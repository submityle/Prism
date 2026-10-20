//! Engine-neutral domain descriptors and editor contribution contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

const MAX_IDENTIFIER_BYTES: usize = 128;
const MAX_TEXT_BYTES: usize = 4 * 1024;

fn validate_identifier(value: &str, kind: &'static str) -> Result<(), ModelError> {
    if value.is_empty() || value.len() > MAX_IDENTIFIER_BYTES {
        return Err(ModelError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    let bytes = value.as_bytes();
    if !bytes[0].is_ascii_lowercase() && !bytes[0].is_ascii_digit() {
        return Err(ModelError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    if !bytes[bytes.len() - 1].is_ascii_lowercase() && !bytes[bytes.len() - 1].is_ascii_digit() {
        return Err(ModelError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    if !bytes
        .iter()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(byte))
    {
        return Err(ModelError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_text(value: &str, field: &'static str) -> Result<(), ModelError> {
    if value.trim().is_empty() {
        return Err(ModelError::EmptyText { field });
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(ModelError::TextTooLong {
            field,
            limit: MAX_TEXT_BYTES,
            actual: value.len(),
        });
    }
    Ok(())
}

macro_rules! string_id {
    ($name:ident, $kind:literal, $docs:literal) => {
        #[doc = $docs]
        #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
        pub struct $name(String);

        impl $name {
            /// Creates an identifier using the canonical lower-case ASCII form.
            pub fn new(value: impl Into<String>) -> Result<Self, ModelError> {
                let value = value.into();
                validate_identifier(&value, $kind)?;
                Ok(Self(value))
            }

            /// Returns the canonical identifier text.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Consumes the identifier and returns its text.
            #[must_use]
            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = ModelError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }
    };
}

string_id!(
    DomainId,
    "domain ID",
    "Stable identity for one editor domain."
);
string_id!(
    CapabilityId,
    "capability ID",
    "Stable identity for a domain capability."
);
string_id!(
    ContributionId,
    "contribution ID",
    "Stable identity for an editor contribution."
);
string_id!(
    ProfileId,
    "profile ID",
    "Stable identity for an optional domain profile."
);

/// Errors raised while constructing an individual domain model value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelError {
    /// An identifier is empty, too long, non-ASCII, or not canonical lower-case.
    InvalidIdentifier {
        /// Identifier category.
        kind: &'static str,
        /// Rejected value.
        value: String,
    },
    /// A human-readable field is empty.
    EmptyText {
        /// Field name.
        field: &'static str,
    },
    /// A human-readable field exceeds the bounded text size.
    TextTooLong {
        /// Field name.
        field: &'static str,
        /// Maximum UTF-8 byte length.
        limit: usize,
        /// Actual UTF-8 byte length.
        actual: usize,
    },
    /// A semantic version string is malformed.
    InvalidVersion(String),
    /// A version range has an inverted or otherwise invalid boundary.
    InvalidVersionRange,
    /// A contribution contains two ports with the same identity.
    DuplicatePort(ContributionId),
    /// A descriptor contains two entries with the same key.
    DuplicateEntry(String),
    /// A profile is owned by a different domain than its descriptor.
    ProfileOwnerMismatch {
        /// Descriptor domain.
        domain: DomainId,
        /// Profile owner.
        profile_domain: DomainId,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { kind, value } => {
                write!(formatter, "invalid {kind}: {value}")
            }
            Self::EmptyText { field } => write!(formatter, "{field} must not be empty"),
            Self::TextTooLong {
                field,
                limit,
                actual,
            } => write!(formatter, "{field} has {actual} bytes; limit is {limit}"),
            Self::InvalidVersion(value) => write!(formatter, "invalid domain version: {value}"),
            Self::InvalidVersionRange => formatter.write_str("invalid version range"),
            Self::DuplicatePort(id) => write!(formatter, "duplicate graph port {id}"),
            Self::DuplicateEntry(value) => write!(formatter, "duplicate descriptor entry: {value}"),
            Self::ProfileOwnerMismatch {
                domain,
                profile_domain,
            } => write!(
                formatter,
                "profile owner {profile_domain} does not match descriptor domain {domain}"
            ),
        }
    }
}

impl std::error::Error for ModelError {}

/// Three-part semantic version used by domains and capabilities.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct DomainVersion {
    /// Incompatible API revision.
    pub major: u32,
    /// Backward-compatible feature revision.
    pub minor: u32,
    /// Bug-fix revision.
    pub patch: u32,
}

impl DomainVersion {
    /// Creates a semantic version.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns whether this version is compatible with a constraint.
    #[must_use]
    pub fn matches(self, constraint: &VersionConstraint) -> bool {
        constraint.matches(self)
    }
}

impl fmt::Display for DomainVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl FromStr for DomainVersion {
    type Err = ModelError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut parts = value.split('.');
        let Some(major) = parts.next() else {
            return Err(ModelError::InvalidVersion(value.to_owned()));
        };
        let Some(minor) = parts.next() else {
            return Err(ModelError::InvalidVersion(value.to_owned()));
        };
        let Some(patch) = parts.next() else {
            return Err(ModelError::InvalidVersion(value.to_owned()));
        };
        if parts.next().is_some() {
            return Err(ModelError::InvalidVersion(value.to_owned()));
        }
        let parse = |part: &str| {
            if part.is_empty() || (part.len() > 1 && part.starts_with('0')) {
                return None;
            }
            part.parse::<u32>().ok()
        };
        match (parse(major), parse(minor), parse(patch)) {
            (Some(major), Some(minor), Some(patch)) => Ok(Self::new(major, minor, patch)),
            _ => Err(ModelError::InvalidVersion(value.to_owned())),
        }
    }
}

/// Inclusive lower and exclusive upper bounds for a semantic version.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct VersionConstraint {
    /// Optional inclusive lower bound.
    pub min_inclusive: Option<DomainVersion>,
    /// Optional exclusive upper bound.
    pub max_exclusive: Option<DomainVersion>,
}

impl VersionConstraint {
    /// Creates and validates a version range.
    pub fn new(
        min_inclusive: Option<DomainVersion>,
        max_exclusive: Option<DomainVersion>,
    ) -> Result<Self, ModelError> {
        if let (Some(min), Some(max)) = (min_inclusive, max_exclusive)
            && min >= max
        {
            return Err(ModelError::InvalidVersionRange);
        }
        Ok(Self {
            min_inclusive,
            max_exclusive,
        })
    }

    /// Creates an unconstrained range.
    #[must_use]
    pub const fn any() -> Self {
        Self {
            min_inclusive: None,
            max_exclusive: None,
        }
    }

    /// Creates a range containing exactly one version.
    pub fn exact(version: DomainVersion) -> Self {
        Self {
            min_inclusive: Some(version),
            max_exclusive: Some(DomainVersion::new(
                version.major,
                version.minor,
                version.patch.saturating_add(1),
            )),
        }
    }

    /// Returns whether a version satisfies this range.
    #[must_use]
    pub fn matches(self, version: DomainVersion) -> bool {
        self.min_inclusive.is_none_or(|min| version >= min)
            && self.max_exclusive.is_none_or(|max| version < max)
    }
}

impl Default for VersionConstraint {
    fn default() -> Self {
        Self::any()
    }
}

/// A capability exposed by a domain.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CapabilityDescriptor {
    /// Stable capability identity.
    pub id: CapabilityId,
    /// Capability protocol version.
    pub version: DomainVersion,
    /// Human-readable capability description.
    pub description: String,
}

impl CapabilityDescriptor {
    /// Creates a capability descriptor.
    pub fn new(
        id: CapabilityId,
        version: DomainVersion,
        description: impl Into<String>,
    ) -> Result<Self, ModelError> {
        let description = description.into();
        validate_text(&description, "capability description")?;
        Ok(Self {
            id,
            version,
            description,
        })
    }
}

/// A dependency on another domain and optionally on its capabilities.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct DomainDependency {
    /// Required domain identity.
    pub domain: DomainId,
    /// Accepted domain version range.
    pub version: VersionConstraint,
    /// Required capability versions keyed by capability identity.
    pub capabilities: BTreeMap<CapabilityId, VersionConstraint>,
}

impl DomainDependency {
    /// Creates an unconstrained dependency.
    #[must_use]
    pub fn new(domain: DomainId) -> Self {
        Self {
            domain,
            version: VersionConstraint::any(),
            capabilities: BTreeMap::new(),
        }
    }

    /// Replaces the accepted domain version range.
    #[must_use]
    pub fn with_version(mut self, version: VersionConstraint) -> Self {
        self.version = version;
        self
    }

    /// Adds a required capability version range.
    #[must_use]
    pub fn requiring_capability(mut self, id: CapabilityId, version: VersionConstraint) -> Self {
        self.capabilities.insert(id, version);
        self
    }
}

/// Severity declared by a validator contribution and used by diagnostics.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum DiagnosticSeverity {
    /// Informational result.
    Info,
    /// Non-fatal issue.
    Warning,
    /// Assembly or lifecycle blocking issue.
    Error,
}

/// Surface on which a panel is placed.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum PanelSurface {
    /// Project-wide primary panel area.
    Main,
    /// Inspector-oriented area.
    Inspector,
    /// Bottom tool area.
    Bottom,
    /// A domain-defined extension surface.
    Extension,
}

/// A render-neutral panel contribution.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PanelContribution {
    /// Stable contribution identity.
    pub id: ContributionId,
    /// Display title.
    pub title: String,
    /// Preferred docking surface.
    pub surface: PanelSurface,
    /// Whether multiple instances may be opened.
    pub multi_instance: bool,
}

impl PanelContribution {
    /// Creates a panel contribution.
    pub fn new(
        id: ContributionId,
        title: impl Into<String>,
        surface: PanelSurface,
    ) -> Result<Self, ModelError> {
        let title = title.into();
        validate_text(&title, "panel title")?;
        Ok(Self {
            id,
            title,
            surface,
            multi_instance: false,
        })
    }

    /// Enables or disables multiple panel instances.
    #[must_use]
    pub fn with_multi_instance(mut self, value: bool) -> Self {
        self.multi_instance = value;
        self
    }
}

/// A render-neutral command contribution.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CommandContribution {
    /// Stable contribution identity.
    pub id: ContributionId,
    /// Display title.
    pub title: String,
    /// Command grouping/category.
    pub category: String,
    /// Whether the command is safe to invoke without a selection.
    pub selection_optional: bool,
}

impl CommandContribution {
    /// Creates a command contribution.
    pub fn new(
        id: ContributionId,
        title: impl Into<String>,
        category: impl Into<String>,
    ) -> Result<Self, ModelError> {
        let title = title.into();
        let category = category.into();
        validate_text(&title, "command title")?;
        validate_text(&category, "command category")?;
        Ok(Self {
            id,
            title,
            category,
            selection_optional: false,
        })
    }

    /// Enables or disables the selection requirement.
    #[must_use]
    pub fn with_selection_optional(mut self, value: bool) -> Self {
        self.selection_optional = value;
        self
    }
}

/// Direction of a graph node port.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum PortDirection {
    /// Input consumed by a node.
    Input,
    /// Output produced by a node.
    Output,
}

/// A typed graph port descriptor.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct GraphPort {
    /// Stable port identity within a node contribution.
    pub id: ContributionId,
    /// Display label.
    pub label: String,
    /// Port direction.
    pub direction: PortDirection,
    /// Type key interpreted by a graph adapter.
    pub value_type: String,
}

impl GraphPort {
    /// Creates a graph port descriptor.
    pub fn new(
        id: ContributionId,
        label: impl Into<String>,
        direction: PortDirection,
        value_type: impl Into<String>,
    ) -> Result<Self, ModelError> {
        let label = label.into();
        let value_type = value_type.into();
        validate_text(&label, "graph port label")?;
        validate_text(&value_type, "graph port value type")?;
        Ok(Self {
            id,
            label,
            direction,
            value_type,
        })
    }
}

/// A render-neutral graph-node contribution.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct GraphNodeContribution {
    /// Stable contribution identity.
    pub id: ContributionId,
    /// Display label.
    pub title: String,
    /// Graph palette/category key.
    pub category: String,
    /// Typed ports exposed to graph adapters.
    pub ports: Vec<GraphPort>,
}

impl GraphNodeContribution {
    /// Creates an empty graph-node contribution.
    pub fn new(
        id: ContributionId,
        title: impl Into<String>,
        category: impl Into<String>,
    ) -> Result<Self, ModelError> {
        let title = title.into();
        let category = category.into();
        validate_text(&title, "graph node title")?;
        validate_text(&category, "graph node category")?;
        Ok(Self {
            id,
            title,
            category,
            ports: Vec::new(),
        })
    }

    /// Adds a port while rejecting duplicate port identities.
    pub fn add_port(&mut self, port: GraphPort) -> Result<(), ModelError> {
        if self.ports.iter().any(|existing| existing.id == port.id) {
            return Err(ModelError::DuplicatePort(port.id));
        }
        self.ports.push(port);
        self.ports.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(())
    }
}

/// A render-neutral timeline contribution.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TimelineContribution {
    /// Stable contribution identity.
    pub id: ContributionId,
    /// Display label.
    pub title: String,
    /// Track type key interpreted by a timeline adapter.
    pub track_type: String,
    /// Whether the track accepts keyframes.
    pub keyframable: bool,
}

impl TimelineContribution {
    /// Creates a timeline contribution.
    pub fn new(
        id: ContributionId,
        title: impl Into<String>,
        track_type: impl Into<String>,
    ) -> Result<Self, ModelError> {
        let title = title.into();
        let track_type = track_type.into();
        validate_text(&title, "timeline title")?;
        validate_text(&track_type, "timeline track type")?;
        Ok(Self {
            id,
            title,
            track_type,
            keyframable: true,
        })
    }

    /// Enables or disables keyframe editing for this track.
    #[must_use]
    pub fn with_keyframable(mut self, value: bool) -> Self {
        self.keyframable = value;
        self
    }
}

/// A render-neutral validator contribution.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ValidatorContribution {
    /// Stable contribution identity.
    pub id: ContributionId,
    /// Display label.
    pub title: String,
    /// Validation scope key interpreted by a validation adapter.
    pub scope: String,
    /// Default severity for emitted findings.
    pub severity: DiagnosticSeverity,
}

impl ValidatorContribution {
    /// Creates a validator contribution.
    pub fn new(
        id: ContributionId,
        title: impl Into<String>,
        scope: impl Into<String>,
        severity: DiagnosticSeverity,
    ) -> Result<Self, ModelError> {
        let title = title.into();
        let scope = scope.into();
        validate_text(&title, "validator title")?;
        validate_text(&scope, "validator scope")?;
        Ok(Self {
            id,
            title,
            scope,
            severity,
        })
    }
}

/// One of the supported editor contribution surfaces.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EditorContribution {
    /// Panel surface contribution.
    Panel(PanelContribution),
    /// Command surface contribution.
    Command(CommandContribution),
    /// Graph-node surface contribution.
    GraphNode(GraphNodeContribution),
    /// Timeline surface contribution.
    Timeline(TimelineContribution),
    /// Validator surface contribution.
    Validator(ValidatorContribution),
}

impl EditorContribution {
    /// Returns the contribution kind.
    #[must_use]
    pub const fn kind(&self) -> ContributionKind {
        match self {
            Self::Panel(_) => ContributionKind::Panel,
            Self::Command(_) => ContributionKind::Command,
            Self::GraphNode(_) => ContributionKind::GraphNode,
            Self::Timeline(_) => ContributionKind::Timeline,
            Self::Validator(_) => ContributionKind::Validator,
        }
    }

    /// Returns the stable contribution identity.
    #[must_use]
    pub fn id(&self) -> &ContributionId {
        match self {
            Self::Panel(value) => &value.id,
            Self::Command(value) => &value.id,
            Self::GraphNode(value) => &value.id,
            Self::Timeline(value) => &value.id,
            Self::Validator(value) => &value.id,
        }
    }
}

/// Contribution surface kind used for collision checks.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum ContributionKind {
    /// Panel contribution.
    Panel,
    /// Command contribution.
    Command,
    /// Graph-node contribution.
    GraphNode,
    /// Timeline contribution.
    Timeline,
    /// Validator contribution.
    Validator,
}

impl fmt::Display for ContributionKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Panel => "panel",
            Self::Command => "command",
            Self::GraphNode => "graph-node",
            Self::Timeline => "timeline",
            Self::Validator => "validator",
        };
        formatter.write_str(value)
    }
}

/// A profile setting value with deterministic ordering and serialization-ready shape.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ProfileValue {
    /// UTF-8 text value.
    Text(String),
    /// Signed integer value.
    Integer(i64),
    /// Boolean value.
    Boolean(bool),
}

/// Optional domain-specific profile registration.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct DomainProfile {
    /// Stable profile identity.
    pub id: ProfileId,
    /// Owning domain identity.
    pub domain: DomainId,
    /// Profile schema version.
    pub schema_version: DomainVersion,
    /// Deterministic key/value settings.
    pub settings: BTreeMap<String, ProfileValue>,
}

impl DomainProfile {
    /// Creates an empty profile owned by a domain.
    #[must_use]
    pub fn new(id: ProfileId, domain: DomainId, schema_version: DomainVersion) -> Self {
        Self {
            id,
            domain,
            schema_version,
            settings: BTreeMap::new(),
        }
    }

    /// Adds or replaces one profile setting.
    pub fn set(&mut self, key: impl Into<String>, value: ProfileValue) -> Result<(), ModelError> {
        let key = key.into();
        validate_identifier(&key, "profile setting key")?;
        self.settings.insert(key, value);
        Ok(())
    }

    /// Validates a profile assembled through public fields.
    pub fn validate(&self) -> Result<(), ModelError> {
        for key in self.settings.keys() {
            validate_identifier(key, "profile setting key")?;
        }
        Ok(())
    }
}

/// A complete descriptor used to assemble one domain into the editor.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct DomainDescriptor {
    /// Stable domain identity.
    pub id: DomainId,
    /// Domain implementation/protocol version.
    pub version: DomainVersion,
    /// Human-readable domain name.
    pub display_name: String,
    /// Capabilities exposed by this domain.
    pub capabilities: BTreeMap<CapabilityId, CapabilityDescriptor>,
    /// Domain dependencies keyed by required domain identity.
    pub dependencies: BTreeMap<DomainId, DomainDependency>,
    /// Editor contributions owned by this domain.
    pub contributions: Vec<EditorContribution>,
    /// Optional profile declaration bundled with this domain.
    pub profile: Option<DomainProfile>,
}

impl DomainDescriptor {
    /// Creates an empty domain descriptor.
    pub fn new(
        id: DomainId,
        version: DomainVersion,
        display_name: impl Into<String>,
    ) -> Result<Self, ModelError> {
        let display_name = display_name.into();
        validate_text(&display_name, "domain display name")?;
        Ok(Self {
            id,
            version,
            display_name,
            capabilities: BTreeMap::new(),
            dependencies: BTreeMap::new(),
            contributions: Vec::new(),
            profile: None,
        })
    }

    /// Adds a capability and rejects a duplicate capability identity.
    pub fn add_capability(&mut self, capability: CapabilityDescriptor) -> Result<(), ModelError> {
        if self.capabilities.contains_key(&capability.id) {
            return Err(ModelError::DuplicateEntry(capability.id.to_string()));
        }
        self.capabilities.insert(capability.id.clone(), capability);
        Ok(())
    }

    /// Adds a dependency and rejects a duplicate dependency identity.
    pub fn add_dependency(&mut self, dependency: DomainDependency) -> Result<(), ModelError> {
        if self.dependencies.contains_key(&dependency.domain) {
            return Err(ModelError::DuplicateEntry(dependency.domain.to_string()));
        }
        self.dependencies
            .insert(dependency.domain.clone(), dependency);
        Ok(())
    }

    /// Adds a contribution and rejects a duplicate surface/identity pair.
    pub fn add_contribution(&mut self, contribution: EditorContribution) -> Result<(), ModelError> {
        if self.contributions.iter().any(|existing| {
            existing.kind() == contribution.kind() && existing.id() == contribution.id()
        }) {
            return Err(ModelError::DuplicateEntry(format!(
                "{}:{}",
                contribution.kind(),
                contribution.id()
            )));
        }
        self.contributions.push(contribution);
        self.contributions.sort_by(|left, right| {
            left.kind()
                .cmp(&right.kind())
                .then_with(|| left.id().cmp(right.id()))
        });
        Ok(())
    }

    /// Sets the optional profile after checking that this descriptor owns it.
    pub fn set_profile(&mut self, profile: DomainProfile) -> Result<(), ModelError> {
        if profile.domain != self.id {
            return Err(ModelError::ProfileOwnerMismatch {
                domain: self.id.clone(),
                profile_domain: profile.domain,
            });
        }
        self.profile = Some(profile);
        Ok(())
    }

    /// Validates a descriptor assembled through public fields.
    pub fn validate(&self) -> Result<(), ModelError> {
        validate_text(&self.display_name, "domain display name")?;
        for (id, capability) in &self.capabilities {
            if id != &capability.id {
                return Err(ModelError::DuplicateEntry(format!(
                    "capability map key {} does not match {}",
                    id, capability.id
                )));
            }
            validate_text(&capability.description, "capability description")?;
        }
        for (id, dependency) in &self.dependencies {
            if id != &dependency.domain {
                return Err(ModelError::DuplicateEntry(format!(
                    "dependency map key {} does not match {}",
                    id, dependency.domain
                )));
            }
        }
        let mut contribution_keys = BTreeSet::new();
        for contribution in &self.contributions {
            let key = (contribution.kind(), contribution.id().clone());
            if !contribution_keys.insert(key.clone()) {
                return Err(ModelError::DuplicateEntry(format!("{}:{}", key.0, key.1)));
            }
            match contribution {
                EditorContribution::Panel(value) => {
                    validate_text(&value.title, "panel title")?;
                }
                EditorContribution::Command(value) => {
                    validate_text(&value.title, "command title")?;
                    validate_text(&value.category, "command category")?;
                }
                EditorContribution::GraphNode(value) => {
                    validate_text(&value.title, "graph node title")?;
                    validate_text(&value.category, "graph node category")?;
                    let mut ports = BTreeSet::new();
                    for port in &value.ports {
                        if !ports.insert(port.id.clone()) {
                            return Err(ModelError::DuplicatePort(port.id.clone()));
                        }
                        validate_text(&port.label, "graph port label")?;
                        validate_text(&port.value_type, "graph port value type")?;
                    }
                }
                EditorContribution::Timeline(value) => {
                    validate_text(&value.title, "timeline title")?;
                    validate_text(&value.track_type, "timeline track type")?;
                }
                EditorContribution::Validator(value) => {
                    validate_text(&value.title, "validator title")?;
                    validate_text(&value.scope, "validator scope")?;
                }
            }
        }
        if let Some(profile) = &self.profile {
            if profile.domain != self.id {
                return Err(ModelError::ProfileOwnerMismatch {
                    domain: self.id.clone(),
                    profile_domain: profile.domain.clone(),
                });
            }
            profile.validate()?;
        }
        Ok(())
    }
}
