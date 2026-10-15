//! Build-time response contracts and compiled-program bindings (02 S1/S9).
//!
//! This module does not implement shader math, composition passes or baking.
//! Field identities come from the caller's schema-generated catalogue; lobe,
//! channel and cache-aux identities come from the existing generated tables.

use crate::{LobeMask, SURFACE_CACHE_AUX, SURFACE_CHANNELS, SURFACE_LOBES};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::{String, ToString},
    vec::Vec,
};
use prism_material_pcm::CapabilitySet;
use prism_render_architecture::shader_package::{
    ArtifactDigest, ArtifactValidationError, ManifestExpectation, PackageState, RegisterError,
    RejectionReason, ShaderPackageId, ShaderPackageManifest, ShaderPackageRegistry,
    ValidationOutcome, VerifiedShaderArtifact,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BuiltinProgram {
    Lit,
    Npr,
    Unlit,
}

/// Registry identity, not a permutation or material-instance identity.
/// Raw construction permits decoding; every registry boundary rejects MAX.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProgramId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ProgramRef {
    Builtin(BuiltinProgram),
    Registered(ProgramId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShadingPlan {
    Single {
        program: ProgramRef,
    },
    MaskBlend {
        left: ProgramRef,
        right: ProgramRef,
        mask: MaskSource,
        domain: BlendDomain,
    },
}

impl Default for ShadingPlan {
    fn default() -> Self {
        Self::Single {
            program: ProgramRef::Builtin(BuiltinProgram::Lit),
        }
    }
}

impl ShadingPlan {
    pub const fn single_builtin(program: BuiltinProgram) -> Self {
        Self::Single {
            program: ProgramRef::Builtin(program),
        }
    }

    pub fn contains_registered(&self) -> bool {
        match self {
            Self::Single { program } => matches!(program, ProgramRef::Registered(_)),
            Self::MaskBlend { left, right, .. } => {
                matches!(left, ProgramRef::Registered(_))
                    || matches!(right, ProgramRef::Registered(_))
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaskSource {
    Texture(String),
    VertexAttr,
    UvChannel,
    RegionMask,
    Channel(String),
}

// These are declarations of the contract, not hand-maintained codegen tables.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlendDomain {
    FullRadiance,
    DirectOnly,
    CustomChannelMask(ChannelMask),
}

pub use prism_render_architecture::shader_package::ExecutionFamily;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct FieldId {
    pub lobe: Option<String>,
    pub name: String,
}

impl FieldId {
    pub fn core(name: impl Into<String>) -> Self {
        Self {
            lobe: None,
            name: name.into(),
        }
    }
    pub fn lobe(lobe: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            lobe: Some(lobe.into()),
            name: name.into(),
        }
    }
}

pub type LobeId = String;
pub type Reason = String;

/// A semantic mapping and a test identifier, not a claim of mathematical parity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldMapping {
    pub field: FieldId,
    pub semantic: String,
    pub test: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SurfaceContract {
    pub required_fields: Vec<FieldId>,
    pub optional_fields: Vec<FieldId>,
    pub ignored_fields_with_reason: Vec<(FieldId, Reason)>,
    pub supported_lobes: Vec<LobeId>,
    pub ignored_lobes_with_reason: Vec<(LobeId, Reason)>,
    pub field_mappings: Vec<FieldMapping>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LightingContract {
    pub direct_light: bool,
    pub shadows: bool,
    pub indirect_diffuse: bool,
    pub reflection: bool,
    pub transmission: bool,
    pub temporal_history: bool,
}

/// Fixed logical ResponseOutput channels. Alpha/auxiliary need S7 merge rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum ResponseChannel {
    Direct,
    IndirectDiffuse,
    Reflection,
    Transmission,
    Emission,
    Alpha,
    Auxiliary,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChannelMask(pub u32);

impl ChannelMask {
    pub const fn one(channel: ResponseChannel) -> Self {
        Self(1 << channel as u8)
    }
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    pub const fn radiance() -> Self {
        Self::one(ResponseChannel::Direct)
            .union(Self::one(ResponseChannel::IndirectDiffuse))
            .union(Self::one(ResponseChannel::Reflection))
            .union(Self::one(ResponseChannel::Transmission))
            .union(Self::one(ResponseChannel::Emission))
    }
    const fn known() -> Self {
        Self::radiance()
            .union(Self::one(ResponseChannel::Alpha))
            .union(Self::one(ResponseChannel::Auxiliary))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputContract {
    pub produced_channels: ChannelMask,
    pub blendable_channels: ChannelMask,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostClass {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionContract {
    pub supports_fused: bool,
    pub supports_split: bool,
    pub max_cost_class: CostClass,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CacheChannel {
    Response(ResponseChannel),
    SurfaceAux(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemporalRequirement {
    None,
    Optional,
    Required,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheContract {
    pub reads: Vec<CacheChannel>,
    pub writes: Vec<CacheChannel>,
    pub temporal: TemporalRequirement,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponseManifest {
    pub surface: SurfaceContract,
    pub lighting: LightingContract,
    pub output: OutputContract,
    pub execution: ExecutionContract,
    pub cache: CacheContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContractError {
    EmptyFieldCatalogue,
    MissingFieldOwner(Option<String>),
    UnknownField(FieldId),
    DuplicateField(FieldId),
    InvalidField(FieldId),
    MissingField(FieldId),
    MissingMapping(FieldId),
    InvalidMapping(FieldId),
    UnknownLobe(String),
    DuplicateLobe(String),
    EmptyReason(String),
    UnsupportedActiveLobe(String),
    UnknownActiveLobes(u32),
    RequiredFieldAbsent(FieldId),
    SupportedLobeWithoutMapping(String),
    MappingOnIgnoredLobe(FieldId),
    InvalidOutput,
    InvalidExecution,
    InvalidTemporal,
    DuplicateCacheChannel(CacheChannel),
    UnknownCacheChannel(CacheChannel),
    CacheWriteNotProduced(CacheChannel),
    MissingLighting(&'static str),
    MissingCacheChannel(CacheChannel),
    UnknownMaskChannel(String),
    EmptyTextureSemantic,
    MissingCapability(String),
    InvalidBlendDomain,
    IncompatibleBlendOutput,
    SplitRequired,
    GeneratedContract(String),
}

/// Identity-only projection of the authoritative, non-padding field directory.
/// The producer must supply *all* fields from schema/codegen, not a hand list.
/// Empty/missing-owner directories are rejected, not treated as full coverage.
#[derive(Clone, Debug)]
pub struct SurfaceFieldCatalog {
    fields: BTreeSet<FieldId>,
}

impl SurfaceFieldCatalog {
    /// Production entry point: derives identities from the generated schema,
    /// never from a second authored list of names or offsets.
    pub fn generated() -> Result<Self, ContractError> {
        Self::new(
            crate::surface_layout::SURFACE_FIELDS
                .iter()
                .map(|field| FieldId {
                    lobe: field.lobe.map(str::to_string),
                    name: field.name.to_string(),
                }),
        )
    }

    pub fn new(fields: impl IntoIterator<Item = FieldId>) -> Result<Self, ContractError> {
        let mut catalogue = BTreeSet::new();
        for field in fields {
            if field.name.trim().is_empty() {
                return Err(ContractError::InvalidField(field));
            }
            if let Some(owner) = &field.lobe {
                lobe_bit(owner)?;
            }
            if !catalogue.insert(field.clone()) {
                return Err(ContractError::DuplicateField(field));
            }
        }
        if catalogue.is_empty() {
            return Err(ContractError::EmptyFieldCatalogue);
        }
        if !catalogue.iter().any(|field| field.lobe.is_none()) {
            return Err(ContractError::MissingFieldOwner(None));
        }
        for lobe in &SURFACE_LOBES {
            if !catalogue
                .iter()
                .any(|field| field.lobe.as_deref() == Some(lobe.id))
            {
                return Err(ContractError::MissingFieldOwner(Some(lobe.id.to_string())));
            }
        }
        Ok(Self { fields: catalogue })
    }
    pub fn fields(&self) -> impl Iterator<Item = &FieldId> {
        self.fields.iter()
    }
}

fn lobe_bit(id: &str) -> Result<u32, ContractError> {
    SURFACE_LOBES
        .iter()
        .find(|lobe| lobe.id == id)
        .map(|lobe| 1 << lobe.registry_slot)
        .ok_or_else(|| ContractError::UnknownLobe(id.to_string()))
}

impl ResponseManifest {
    /// Build a built-in response manifest from the schema-generated contract
    /// table. No field, lobe, channel or cache identity is duplicated here.
    pub fn from_builtin(program: BuiltinProgram) -> Result<Self, ContractError> {
        let builtin = match program {
            BuiltinProgram::Lit => "lit",
            BuiltinProgram::Npr => "npr",
            BuiltinProgram::Unlit => "unlit",
        };
        let layout = crate::SURFACE_RESPONSE_PROGRAMS
            .iter()
            .find(|layout| layout.builtin == builtin)
            .ok_or_else(|| ContractError::GeneratedContract(builtin.to_string()))?;

        let channel = |name: &str| match name {
            "direct" => Ok(ResponseChannel::Direct),
            "indirect_diffuse" => Ok(ResponseChannel::IndirectDiffuse),
            "reflection" => Ok(ResponseChannel::Reflection),
            "transmission" => Ok(ResponseChannel::Transmission),
            "emission" => Ok(ResponseChannel::Emission),
            "alpha" => Ok(ResponseChannel::Alpha),
            "auxiliary" => Ok(ResponseChannel::Auxiliary),
            _ => Err(ContractError::GeneratedContract(name.to_string())),
        };
        let channels = |names: &[&str]| {
            names.iter().try_fold(ChannelMask::default(), |mask, name| {
                Ok(mask.union(ChannelMask::one(channel(name)?)))
            })
        };
        let cache_channels = |names: &[&str]| {
            names
                .iter()
                .map(|name| Ok(CacheChannel::Response(channel(name)?)))
                .collect::<Result<Vec<_>, ContractError>>()
        };
        let cost = match layout.max_cost_class {
            "low" => CostClass::Low,
            "medium" => CostClass::Medium,
            "high" => CostClass::High,
            other => return Err(ContractError::GeneratedContract(other.to_string())),
        };
        let temporal = match layout.cache_temporal {
            "none" => TemporalRequirement::None,
            "optional" => TemporalRequirement::Optional,
            "required" => TemporalRequirement::Required,
            other => return Err(ContractError::GeneratedContract(other.to_string())),
        };
        let mut surface = SurfaceContract::default();
        surface.supported_lobes = layout
            .lobes
            .iter()
            .filter(|lobe| lobe.supported)
            .map(|lobe| lobe.id.to_string())
            .collect();
        surface.ignored_lobes_with_reason = layout
            .lobes
            .iter()
            .filter_map(|lobe| {
                lobe.ignored_reason
                    .map(|reason| (lobe.id.to_string(), reason.to_string()))
            })
            .collect();
        surface.optional_fields = layout
            .field_mappings
            .iter()
            .map(|field| FieldId {
                lobe: field.lobe.map(str::to_string),
                name: field.name.to_string(),
            })
            .collect();
        surface.ignored_fields_with_reason = layout
            .ignored_fields
            .iter()
            .map(|field| {
                (
                    FieldId {
                        lobe: field.lobe.map(str::to_string),
                        name: field.name.to_string(),
                    },
                    field.reason.to_string(),
                )
            })
            .collect();
        surface.field_mappings = layout
            .field_mappings
            .iter()
            .map(|field| FieldMapping {
                field: FieldId {
                    lobe: field.lobe.map(str::to_string),
                    name: field.name.to_string(),
                },
                semantic: field.semantic.to_string(),
                test: field.test.to_string(),
            })
            .collect();

        Ok(Self {
            surface,
            lighting: LightingContract {
                direct_light: layout.direct_light,
                shadows: layout.shadows,
                indirect_diffuse: layout.indirect_diffuse,
                reflection: layout.reflection,
                transmission: layout.transmission,
                temporal_history: layout.temporal_history,
            },
            output: OutputContract {
                produced_channels: channels(layout.produced_channels)?,
                blendable_channels: channels(layout.blendable_channels)?,
            },
            execution: ExecutionContract {
                supports_fused: layout.supports_fused,
                supports_split: layout.supports_split,
                max_cost_class: cost,
            },
            cache: CacheContract {
                reads: cache_channels(layout.cache_reads)?,
                writes: cache_channels(layout.cache_writes)?,
                temporal,
            },
        })
    }

    /// Checks all declarations, including fields belonging to inactive lobes.
    pub fn validate(&self, fields: &SurfaceFieldCatalog) -> Result<(), ContractError> {
        let surface = &self.surface;
        let mut declarations = BTreeSet::new();
        let mapped: BTreeSet<_> = surface
            .required_fields
            .iter()
            .chain(&surface.optional_fields)
            .collect();
        for field in surface
            .required_fields
            .iter()
            .chain(&surface.optional_fields)
            .chain(
                surface
                    .ignored_fields_with_reason
                    .iter()
                    .map(|(field, _)| field),
            )
        {
            if !fields.fields.contains(field) {
                return Err(ContractError::UnknownField(field.clone()));
            }
            if !declarations.insert(field) {
                return Err(ContractError::DuplicateField(field.clone()));
            }
        }
        for (field, reason) in &surface.ignored_fields_with_reason {
            if reason.trim().is_empty() {
                return Err(ContractError::EmptyReason(field.name.clone()));
            }
        }
        for field in &fields.fields {
            if !declarations.contains(field) {
                return Err(ContractError::MissingField(field.clone()));
            }
        }
        let mut mappings = BTreeSet::new();
        for mapping in &surface.field_mappings {
            if !mapped.contains(&mapping.field)
                || mapping.semantic.trim().is_empty()
                || mapping.test.trim().is_empty()
                || !mappings.insert(&mapping.field)
            {
                return Err(ContractError::InvalidMapping(mapping.field.clone()));
            }
        }
        for field in mapped {
            if !mappings.contains(field) {
                return Err(ContractError::MissingMapping(field.clone()));
            }
        }
        let mut lobes = BTreeSet::new();
        for lobe in surface.supported_lobes.iter().chain(
            surface
                .ignored_lobes_with_reason
                .iter()
                .map(|(lobe, _)| lobe),
        ) {
            lobe_bit(lobe)?;
            if !lobes.insert(lobe) {
                return Err(ContractError::DuplicateLobe(lobe.clone()));
            }
        }
        for (lobe, reason) in &surface.ignored_lobes_with_reason {
            if reason.trim().is_empty() {
                return Err(ContractError::EmptyReason(lobe.clone()));
            }
            if let Some(field) = mappings
                .iter()
                .find(|field| field.lobe.as_deref() == Some(lobe))
            {
                return Err(ContractError::MappingOnIgnoredLobe((*field).clone()));
            }
        }
        for lobe in &surface.supported_lobes {
            if !mappings
                .iter()
                .any(|field| field.lobe.as_deref() == Some(lobe))
            {
                return Err(ContractError::SupportedLobeWithoutMapping(lobe.clone()));
            }
        }
        let output = self.output;
        if output.produced_channels.0 == 0
            || !ChannelMask::known().contains(output.produced_channels)
            || !output.produced_channels.contains(output.blendable_channels)
        {
            return Err(ContractError::InvalidOutput);
        }
        if !self.execution.supports_fused && !self.execution.supports_split {
            return Err(ContractError::InvalidExecution);
        }
        if self.lighting.temporal_history && self.cache.temporal == TemporalRequirement::None {
            return Err(ContractError::InvalidTemporal);
        }
        for channels in [&self.cache.reads, &self.cache.writes] {
            let mut seen = BTreeSet::new();
            for channel in channels {
                if !seen.insert(channel) {
                    return Err(ContractError::DuplicateCacheChannel(channel.clone()));
                }
                if let CacheChannel::SurfaceAux(id) = channel
                    && !SURFACE_CACHE_AUX.iter().any(|aux| aux.id == id)
                {
                    return Err(ContractError::UnknownCacheChannel(channel.clone()));
                }
            }
        }
        for channel in &self.cache.writes {
            if let CacheChannel::Response(channel_id) = channel
                && !output
                    .produced_channels
                    .contains(ChannelMask::one(*channel_id))
            {
                return Err(ContractError::CacheWriteNotProduced(channel.clone()));
            }
        }
        Ok(())
    }

    pub fn validate_for_surface(
        &self,
        fields: &SurfaceFieldCatalog,
        active_lobes: LobeMask,
    ) -> Result<(), ContractError> {
        self.validate(fields)?;
        let unknown = active_lobes.bits() & !LobeMask::all().bits();
        if unknown != 0 {
            return Err(ContractError::UnknownActiveLobes(unknown));
        }
        for lobe in &SURFACE_LOBES {
            if active_lobes.bits() & (1 << lobe.registry_slot) != 0
                && !self.surface.supported_lobes.iter().any(|id| id == lobe.id)
                && !self
                    .surface
                    .ignored_lobes_with_reason
                    .iter()
                    .any(|(id, _)| id == lobe.id)
            {
                return Err(ContractError::UnsupportedActiveLobe(lobe.id.to_string()));
            }
        }
        for field in &self.surface.required_fields {
            if let Some(lobe) = &field.lobe
                && active_lobes.bits() & lobe_bit(lobe)? == 0
            {
                return Err(ContractError::RequiredFieldAbsent(field.clone()));
            }
        }
        Ok(())
    }
}

/// Supplied build resources, not per-frame quality switches or permutation bits.
#[derive(Clone, Debug)]
pub struct PlanValidationContext {
    pub active_lobes: LobeMask,
    pub capabilities: CapabilitySet,
    pub lighting_inputs: LightingContract,
    pub cache_channels: Vec<CacheChannel>,
    pub temporal_history_available: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledEntryError {
    ResponseSource(String),
    InvalidHeader,
    MalformedInstruction(usize),
    InvalidEntryName,
    DuplicateEntry,
    EntryMissing(String),
    FunctionMissing(u32),
    InvalidFunction(u32),
    EmptyProvenance,
    Validation(ArtifactValidationError),
}

/// Immutable SPIR-V artifact plus an entry actually declared and defined in it.
/// Full Naga validation precedes binding. Device support and response semantics
/// are separately checked by pipeline creation and response conformance tests.
#[derive(Clone, Debug)]
pub struct CompiledEntry {
    manifest: ShaderPackageManifest,
    artifact: VerifiedShaderArtifact,
    function_id: u32,
    pub(crate) response_source: Option<crate::response_source::ResponseSource>,
}

impl CompiledEntry {
    pub fn from_spirv(
        manifest: ShaderPackageManifest,
        spirv: Vec<u32>,
    ) -> Result<Self, CompiledEntryError> {
        if manifest.id.as_str().trim().is_empty()
            || manifest.source_revision.trim().is_empty()
            || manifest.entry_point.trim().is_empty()
        {
            return Err(CompiledEntryError::EmptyProvenance);
        }
        if spirv.len() < 5
            || spirv[0] != 0x0723_0203
            || spirv[3] == 0
            || spirv[4] != 0
            || spirv[1] & 0xff != 0
            || (spirv[1] >> 16) != 1
            || ((spirv[1] >> 8) & 0xff) > 6
        {
            return Err(CompiledEntryError::InvalidHeader);
        }
        let mut cursor = 5;
        let mut entries = BTreeMap::new();
        let mut functions = BTreeSet::new();
        let mut current_function = None;
        let mut has_label = false;
        while cursor < spirv.len() {
            let count = (spirv[cursor] >> 16) as usize;
            let opcode = spirv[cursor] & 0xffff;
            if count == 0 || count > spirv.len() - cursor {
                return Err(CompiledEntryError::MalformedInstruction(cursor));
            }
            let instruction = &spirv[cursor..cursor + count];
            match opcode {
                15 => {
                    // OpEntryPoint: execution model, function id, literal string, interfaces.
                    if count < 4 || instruction[2] == 0 || instruction[2] >= spirv[3] {
                        return Err(CompiledEntryError::MalformedInstruction(cursor));
                    }
                    let mut bytes = Vec::new();
                    let mut terminated = false;
                    for word in &instruction[3..] {
                        for byte in word.to_le_bytes() {
                            if byte == 0 {
                                terminated = true;
                                break;
                            }
                            bytes.push(byte);
                        }
                        if terminated {
                            break;
                        }
                    }
                    if !terminated {
                        return Err(CompiledEntryError::InvalidEntryName);
                    }
                    let name = String::from_utf8(bytes)
                        .map_err(|_| CompiledEntryError::InvalidEntryName)?;
                    if name.is_empty() {
                        return Err(CompiledEntryError::InvalidEntryName);
                    }
                    if entries.insert(name, instruction[2]).is_some() {
                        return Err(CompiledEntryError::DuplicateEntry);
                    }
                }
                54 => {
                    // OpFunction: result type, result id, control, function type.
                    if count != 5
                        || current_function.is_some()
                        || instruction[2] == 0
                        || instruction[2] >= spirv[3]
                        || functions.contains(&instruction[2])
                    {
                        return Err(CompiledEntryError::MalformedInstruction(cursor));
                    }
                    current_function = Some(instruction[2]);
                    has_label = false;
                }
                248 => {
                    // OpLabel: function declarations alone are not compiled definitions.
                    if count != 2 || current_function.is_none() {
                        return Err(CompiledEntryError::MalformedInstruction(cursor));
                    }
                    has_label = true;
                }
                56 => {
                    // OpFunctionEnd
                    if count != 1 {
                        return Err(CompiledEntryError::MalformedInstruction(cursor));
                    }
                    let id = current_function
                        .take()
                        .ok_or(CompiledEntryError::MalformedInstruction(cursor))?;
                    if !has_label {
                        return Err(CompiledEntryError::InvalidFunction(id));
                    }
                    functions.insert(id);
                }
                _ => {}
            }
            cursor += count;
        }
        if let Some(id) = current_function {
            return Err(CompiledEntryError::InvalidFunction(id));
        }
        let function_id = entries
            .get(&manifest.entry_point)
            .copied()
            .ok_or_else(|| CompiledEntryError::EntryMissing(manifest.entry_point.clone()))?;
        if !functions.contains(&function_id) {
            return Err(CompiledEntryError::FunctionMissing(function_id));
        }
        let artifact = VerifiedShaderArtifact::from_spirv(manifest.clone(), spirv)
            .map_err(CompiledEntryError::Validation)?;
        Ok(Self {
            manifest,
            artifact,
            function_id,
            response_source: None,
        })
    }
    pub fn package_manifest(&self) -> &ShaderPackageManifest {
        &self.manifest
    }
    pub fn spirv(&self) -> &[u32] {
        self.artifact.spirv()
    }
    pub fn artifact(&self) -> &VerifiedShaderArtifact {
        &self.artifact
    }
    pub fn digest(&self) -> ArtifactDigest {
        self.artifact.digest()
    }
    pub fn function_id(&self) -> u32 {
        self.function_id
    }
    /// Validated function bodies available for linking into the scene resolve.
    /// A SPIR-V-only package has no implicit source or hash-based substitute.
    pub fn response_source(&self) -> Option<&crate::response_source::ResponseSource> {
        self.response_source.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RegistryError {
    ReservedProgramId,
    DuplicateProgram(ProgramRef),
    RetiredProgramId(ProgramId),
    StaleProgram(ProgramRef),
    PackageUnavailable,
    PackageChanged,
    PackageNotVerified,
    PackageCodeChanged,
    PackageRegistration(RegisterError),
    Compile(CompiledEntryError),
    PackageAbi(RejectionReason),
    Contract(ContractError),
    CustomRequiresSplit,
}

impl From<ContractError> for RegistryError {
    fn from(error: ContractError) -> Self {
        Self::Contract(error)
    }
}

impl From<CompiledEntryError> for RegistryError {
    fn from(error: CompiledEntryError) -> Self {
        Self::Compile(error)
    }
}

impl From<RegisterError> for RegistryError {
    fn from(error: RegisterError) -> Self {
        Self::PackageRegistration(error)
    }
}

#[derive(Clone, Debug)]
pub struct RegisteredResponse {
    manifest: ResponseManifest,
    compiled: CompiledEntry,
}

/// Receipt returned by the end-to-end custom response producer.
#[derive(Clone, Debug)]
pub struct CustomResponseReceipt {
    /// Stable response-program identity used by authoring and material data.
    pub program: ProgramRef,
    /// Package identity used by the verified shader registry.
    pub package: ShaderPackageId,
    /// Content digest of the accepted executable bytes.
    pub digest: ArtifactDigest,
    /// Reflected entry-point function ID in the executable module.
    pub function_id: u32,
}

impl RegisteredResponse {
    pub fn manifest(&self) -> &ResponseManifest {
        &self.manifest
    }
    pub fn compiled_entry(&self) -> &CompiledEntry {
        &self.compiled
    }
}

/// A validated build-time choice. Borrowing prevents mutation/invalidation of
/// the response registry until the baker has consumed both bindings.
#[derive(Debug)]
pub struct ValidatedPlan<'a> {
    pub execution_family: ExecutionFamily,
    pub left: &'a RegisteredResponse,
    pub right: Option<&'a RegisteredResponse>,
    pub blended_channels: ChannelMask,
}

#[derive(Clone, Debug)]
pub struct ResponseRegistry {
    fields: SurfaceFieldCatalog,
    abi_expectation: ManifestExpectation,
    programs: BTreeMap<ProgramRef, RegisteredResponse>,
    retired: BTreeSet<ProgramId>,
}

impl ResponseRegistry {
    pub fn new(fields: SurfaceFieldCatalog, abi_expectation: ManifestExpectation) -> Self {
        Self {
            fields,
            abi_expectation,
            programs: BTreeMap::new(),
            retired: BTreeSet::new(),
        }
    }

    /// Produces and registers a custom response from executable SPIR-V bytes.
    ///
    /// This is the single cold-path entry for custom programs: reflection and
    /// Naga validation happen before the package is made ready, the package is
    /// checked against this registry's ABI expectation, and only then is the
    /// registered `ProgramRef` published. A stale or reserved ID is rejected;
    /// no builtin or default program is substituted.
    pub fn register_custom_spirv(
        &mut self,
        program_id: ProgramId,
        manifest: ResponseManifest,
        package_manifest: ShaderPackageManifest,
        spirv: Vec<u32>,
        packages: &mut ShaderPackageRegistry,
    ) -> Result<CustomResponseReceipt, RegistryError> {
        let program = ProgramRef::Registered(program_id);
        check_id(program)?;
        if self.programs.contains_key(&program) {
            return Err(RegistryError::DuplicateProgram(program));
        }
        if self.retired.contains(&program_id) {
            return Err(RegistryError::RetiredProgramId(program_id));
        }
        if manifest.execution.supports_fused || !manifest.execution.supports_split {
            return Err(RegistryError::CustomRequiresSplit);
        }
        manifest.validate(&self.fields)?;

        let compiled = CompiledEntry::from_spirv(package_manifest, spirv)?;
        let package_id = compiled.package_manifest().id.clone();
        if let Some(existing) = packages.verified_artifact(&package_id) {
            if existing.digest() != compiled.digest() {
                return Err(RegistryError::PackageCodeChanged);
            }
        } else if packages.contains(&package_id) {
            return Err(RegistryError::PackageNotVerified);
        } else {
            packages.register_verified(compiled.artifact().clone())?;
        }
        let outcome = packages
            .validate(&package_id, &self.abi_expectation)
            .ok_or(RegistryError::PackageUnavailable)?;
        if let ValidationOutcome::Rejected(reason) = outcome {
            return Err(RegistryError::PackageAbi(reason));
        }
        self.check_binding(&compiled, packages)?;
        self.register(program, manifest, compiled.clone(), packages)?;
        Ok(CustomResponseReceipt {
            program,
            package: package_id,
            digest: compiled.digest(),
            function_id: compiled.function_id(),
        })
    }
    /// Atomic: no id is consumed if any contract/artifact/package check fails.
    pub fn register(
        &mut self,
        program: ProgramRef,
        manifest: ResponseManifest,
        compiled: CompiledEntry,
        packages: &ShaderPackageRegistry,
    ) -> Result<(), RegistryError> {
        check_id(program)?;
        if self.programs.contains_key(&program) {
            return Err(RegistryError::DuplicateProgram(program));
        }
        if let ProgramRef::Registered(id) = program {
            if self.retired.contains(&id) {
                return Err(RegistryError::RetiredProgramId(id));
            }
            if manifest.execution.supports_fused || !manifest.execution.supports_split {
                return Err(RegistryError::CustomRequiresSplit);
            }
        }
        manifest.validate(&self.fields)?;
        self.check_binding(&compiled, packages)?;
        self.programs
            .insert(program, RegisteredResponse { manifest, compiled });
        Ok(())
    }
    pub fn resolve(
        &self,
        program: ProgramRef,
        packages: &ShaderPackageRegistry,
    ) -> Result<&RegisteredResponse, RegistryError> {
        check_id(program)?;
        let entry = self
            .programs
            .get(&program)
            .ok_or(RegistryError::StaleProgram(program))?;
        self.check_binding(&entry.compiled, packages)?;
        Ok(entry)
    }
    /// Removed numeric IDs are never reused, avoiding stale-material aliasing.
    pub fn unregister(&mut self, program: ProgramRef) -> Result<(), RegistryError> {
        check_id(program)?;
        self.programs
            .remove(&program)
            .ok_or(RegistryError::StaleProgram(program))?;
        if let ProgramRef::Registered(id) = program {
            self.retired.insert(id);
        }
        Ok(())
    }
    pub fn fields(&self) -> &SurfaceFieldCatalog {
        &self.fields
    }
    pub fn len(&self) -> usize {
        self.programs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.programs.is_empty()
    }

    fn check_binding(
        &self,
        compiled: &CompiledEntry,
        packages: &ShaderPackageRegistry,
    ) -> Result<(), RegistryError> {
        let expected = &compiled.manifest;
        let package = packages
            .get(&expected.id)
            .ok_or(RegistryError::PackageUnavailable)?;
        if package.state != PackageState::Ready {
            return Err(RegistryError::PackageUnavailable);
        }
        let actual = &package.manifest;
        if actual.id != expected.id
            || actual.source_revision != expected.source_revision
            || actual.entry_point != expected.entry_point
            || actual.abi_version != expected.abi_version
            || actual.abi_hash != expected.abi_hash
            || actual.permutation_count != expected.permutation_count
        {
            return Err(RegistryError::PackageChanged);
        }
        if let ValidationOutcome::Rejected(reason) = actual.validate(&self.abi_expectation) {
            return Err(RegistryError::PackageAbi(reason));
        }
        let artifact = packages
            .verified_artifact(&expected.id)
            .ok_or(RegistryError::PackageNotVerified)?;
        if artifact.digest() != compiled.digest() {
            return Err(RegistryError::PackageCodeChanged);
        }
        Ok(())
    }

    pub fn validate_plan<'a>(
        &'a self,
        plan: &ShadingPlan,
        context: &PlanValidationContext,
        packages: &ShaderPackageRegistry,
    ) -> Result<ValidatedPlan<'a>, RegistryError> {
        match plan {
            ShadingPlan::Single { program } => {
                let entry = self.validate_arm(*program, context, packages)?;
                let execution_family = if matches!(program, ProgramRef::Builtin(_))
                    && entry.manifest.execution.supports_fused
                {
                    ExecutionFamily::FusedBuiltin
                } else {
                    ExecutionFamily::SplitResponse
                };
                Ok(ValidatedPlan {
                    execution_family,
                    left: entry,
                    right: None,
                    blended_channels: ChannelMask::default(),
                })
            }
            ShadingPlan::MaskBlend {
                left,
                right,
                mask,
                domain,
            } => {
                let left_entry = self.validate_arm(*left, context, packages)?;
                let right_entry = self.validate_arm(*right, context, packages)?;
                validate_mask(mask, context)?;
                let blended_channels = match domain {
                    BlendDomain::FullRadiance => ChannelMask::radiance(),
                    BlendDomain::DirectOnly => ChannelMask::one(ResponseChannel::Direct),
                    BlendDomain::CustomChannelMask(channels) => {
                        if channels.0 == 0 || !ChannelMask::radiance().contains(*channels) {
                            return Err(ContractError::InvalidBlendDomain.into());
                        }
                        *channels
                    }
                };
                for entry in [left_entry, right_entry] {
                    if !entry
                        .manifest
                        .output
                        .produced_channels
                        .contains(blended_channels)
                        || !entry
                            .manifest
                            .output
                            .blendable_channels
                            .contains(blended_channels)
                    {
                        return Err(ContractError::IncompatibleBlendOutput.into());
                    }
                }
                let fused = matches!(left, ProgramRef::Builtin(_))
                    && matches!(right, ProgramRef::Builtin(_))
                    && left_entry.manifest.execution.supports_fused
                    && right_entry.manifest.execution.supports_fused;
                if !fused
                    && (!left_entry.manifest.execution.supports_split
                        || !right_entry.manifest.execution.supports_split)
                {
                    return Err(ContractError::SplitRequired.into());
                }
                Ok(ValidatedPlan {
                    execution_family: if fused {
                        ExecutionFamily::FusedBuiltin
                    } else {
                        ExecutionFamily::Composition
                    },
                    left: left_entry,
                    right: Some(right_entry),
                    blended_channels,
                })
            }
        }
    }

    fn validate_arm(
        &self,
        program: ProgramRef,
        context: &PlanValidationContext,
        packages: &ShaderPackageRegistry,
    ) -> Result<&RegisteredResponse, RegistryError> {
        let entry = self.resolve(program, packages)?;
        entry
            .manifest
            .validate_for_surface(&self.fields, context.active_lobes)?;
        let needs = entry.manifest.lighting;
        let available = context.lighting_inputs;
        for (name, needed, present) in [
            ("direct_light", needs.direct_light, available.direct_light),
            ("shadows", needs.shadows, available.shadows),
            (
                "indirect_diffuse",
                needs.indirect_diffuse,
                available.indirect_diffuse,
            ),
            ("reflection", needs.reflection, available.reflection),
            ("transmission", needs.transmission, available.transmission),
        ] {
            if needed && !present {
                return Err(ContractError::MissingLighting(name).into());
            }
        }
        if (entry.manifest.cache.temporal == TemporalRequirement::Required
            || needs.temporal_history)
            && (!context.temporal_history_available || !available.temporal_history)
        {
            return Err(ContractError::MissingLighting("temporal_history").into());
        }
        for channel in &entry.manifest.cache.reads {
            if !context.cache_channels.contains(channel) {
                return Err(ContractError::MissingCacheChannel(channel.clone()).into());
            }
        }
        Ok(entry)
    }
}

fn check_id(program: ProgramRef) -> Result<(), RegistryError> {
    if program == ProgramRef::Registered(ProgramId(u32::MAX)) {
        return Err(RegistryError::ReservedProgramId);
    }
    Ok(())
}

fn validate_mask(mask: &MaskSource, context: &PlanValidationContext) -> Result<(), ContractError> {
    match mask {
        MaskSource::Texture(semantic) if semantic.trim().is_empty() => {
            Err(ContractError::EmptyTextureSemantic)
        }
        MaskSource::Channel(id) => {
            let channel = SURFACE_CHANNELS
                .iter()
                .find(|channel| channel.id == id)
                .ok_or_else(|| ContractError::UnknownMaskChannel(id.clone()))?;
            for name in channel.requires_caps {
                let capability = crate::capability::Capability::parse(name)
                    .ok_or_else(|| ContractError::MissingCapability((*name).to_string()))?;
                if !context.capabilities.contains(capability) {
                    return Err(ContractError::MissingCapability((*name).to_string()));
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
