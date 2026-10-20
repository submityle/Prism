use crate::clock::HlcTimestamp;
use core::fmt;
use std::collections::BTreeSet;

/// Stable identity of a collaboration actor.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ActorId(String);

impl ActorId {
    /// Creates an actor identity and rejects empty values.
    pub fn new(value: impl Into<String>) -> Result<Self, CommandError> {
        let value = value.into();
        if value.is_empty() {
            return Err(CommandError::EmptyIdentifier("actor"));
        }
        Ok(Self(value))
    }

    /// Creates an actor identity for trusted static data.
    #[must_use]
    pub fn from_static(value: &'static str) -> Self {
        assert!(
            !value.is_empty(),
            "static actor identifiers must not be empty"
        );
        Self(value.to_owned())
    }

    /// Returns the canonical actor text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ActorId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Domain touched by a command, such as a document or asset namespace.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct DomainId(String);

impl DomainId {
    /// Creates a non-empty domain identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, CommandError> {
        let value = value.into();
        if value.is_empty() {
            return Err(CommandError::EmptyIdentifier("domain"));
        }
        Ok(Self(value))
    }

    /// Creates a domain identifier for trusted static data.
    #[must_use]
    pub fn from_static(value: &'static str) -> Self {
        assert!(
            !value.is_empty(),
            "static domain identifiers must not be empty"
        );
        Self(value.to_owned())
    }

    /// Returns the canonical domain text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stable identity of one command in an actor sequence.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct CommandId {
    /// Actor that emitted the command.
    pub actor: ActorId,
    /// Actor-local monotonic sequence.
    pub sequence: u64,
}

/// A path segment in a field-level register key.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum PathSegment {
    /// Named field.
    Field(String),
    /// String-keyed map entry.
    Key(String),
    /// Numeric list/index entry.
    Index(u32),
}

/// Canonical path used by the field-level LWW map.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct FieldPath(Vec<PathSegment>);

impl FieldPath {
    /// Creates a non-empty field path and validates named segments.
    pub fn new(segments: Vec<PathSegment>) -> Result<Self, CommandError> {
        if segments.is_empty() {
            return Err(CommandError::EmptyIdentifier("field path"));
        }
        for segment in &segments {
            if matches!(segment, PathSegment::Field(value) | PathSegment::Key(value) if value.is_empty())
            {
                return Err(CommandError::EmptyIdentifier("field path segment"));
            }
        }
        Ok(Self(segments))
    }

    /// Creates a one-field path.
    pub fn field(value: impl Into<String>) -> Result<Self, CommandError> {
        Self::new(vec![PathSegment::Field(value.into())])
    }

    /// Returns the path segments.
    #[must_use]
    pub fn segments(&self) -> &[PathSegment] {
        &self.0
    }

    /// Returns a deterministic display form.
    #[must_use]
    pub fn as_string(&self) -> String {
        let mut output = String::new();
        for segment in &self.0 {
            match segment {
                PathSegment::Field(value) => {
                    output.push('.');
                    output.push_str(value);
                }
                PathSegment::Key(value) => {
                    output.push_str("[\"");
                    output.push_str(value);
                    output.push_str("\"]");
                }
                PathSegment::Index(value) => {
                    output.push('[');
                    output.push_str(&value.to_string());
                    output.push(']');
                }
            }
        }
        output.trim_start_matches('.').to_owned()
    }
}

/// Scalar values supported by the generic field register.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Value {
    /// Explicit null.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Signed integer.
    Int(i64),
    /// Unsigned integer.
    UInt(u64),
    /// Exact IEEE-754 bits.
    FloatBits(u64),
    /// UTF-8 text.
    Text(String),
    /// Opaque bytes for adapter-defined scalar values.
    Bytes(Vec<u8>),
}

impl Value {
    /// Stores an `f64` without normalizing its bit pattern.
    #[must_use]
    pub fn float(value: f64) -> Self {
        Self::FloatBits(value.to_bits())
    }
}

/// Payload carried by a command envelope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandPayload {
    /// Sets one field register.
    SetField {
        /// Field register path.
        path: FieldPath,
        /// New register value.
        value: Value,
    },
    /// Writes a tombstone for one field register.
    DeleteField {
        /// Field register path.
        path: FieldPath,
    },
    /// Adapter-defined command bytes. The replica records these but does not
    /// interpret them.
    Opaque(Vec<u8>),
}

/// Limits checked at the command boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommandLimits {
    /// Maximum number of affected domains.
    pub max_domains: usize,
    /// Maximum number of causal references.
    pub max_causal_dependencies: usize,
    /// Maximum opaque payload/value byte length.
    pub max_payload_bytes: usize,
}

impl Default for CommandLimits {
    fn default() -> Self {
        Self {
            max_domains: 64,
            max_causal_dependencies: 256,
            max_payload_bytes: 4 * 1024 * 1024,
        }
    }
}

/// Command construction and validation failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandError {
    /// An identifier or path segment was empty.
    EmptyIdentifier(&'static str),
    /// Sequence zero is reserved for the initial actor state.
    InvalidSequence,
    /// The envelope actor and command ID actor differ.
    ActorMismatch,
    /// The envelope has no affected domain.
    NoAffectedDomains,
    /// A collection contains duplicate entries.
    DuplicateEntry(&'static str),
    /// A collection exceeds its configured limit.
    LimitExceeded(&'static str),
    /// The envelope references itself causally.
    SelfDependency,
}

impl fmt::Display for CommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid collaboration command: {self:?}")
    }
}

impl std::error::Error for CommandError {}

/// A transport-neutral command with causal metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandEnvelope {
    /// Stable command identity.
    pub id: CommandId,
    /// Actor that authored the command.
    pub actor: ActorId,
    /// HLC timestamp assigned by the author.
    pub timestamp: HlcTimestamp,
    /// Commands that must be applied before this command.
    pub causal: Vec<CommandId>,
    /// Domains affected by the command.
    pub affected_domains: BTreeSet<DomainId>,
    /// Operation or adapter-defined bytes.
    pub payload: CommandPayload,
}

impl CommandEnvelope {
    /// Creates and validates an envelope.
    pub fn new(
        actor: ActorId,
        sequence: u64,
        timestamp: HlcTimestamp,
        causal: Vec<CommandId>,
        affected_domains: BTreeSet<DomainId>,
        payload: CommandPayload,
        limits: CommandLimits,
    ) -> Result<Self, CommandError> {
        if sequence == 0 {
            return Err(CommandError::InvalidSequence);
        }
        if affected_domains.is_empty() {
            return Err(CommandError::NoAffectedDomains);
        }
        if affected_domains.len() > limits.max_domains {
            return Err(CommandError::LimitExceeded("affected domains"));
        }
        if causal.len() > limits.max_causal_dependencies {
            return Err(CommandError::LimitExceeded("causal dependencies"));
        }
        let mut normalized_causal = causal;
        normalized_causal.sort();
        let had_duplicate = normalized_causal.windows(2).any(|pair| pair[0] == pair[1]);
        if had_duplicate {
            return Err(CommandError::DuplicateEntry("causal dependencies"));
        }
        let id = CommandId {
            actor: actor.clone(),
            sequence,
        };
        if normalized_causal.iter().any(|dependency| dependency == &id) {
            return Err(CommandError::SelfDependency);
        }
        let payload_size = match &payload {
            CommandPayload::Opaque(bytes) => bytes.len(),
            CommandPayload::SetField { path, value } => path.as_string().len() + value_size(value),
            CommandPayload::DeleteField { path } => path.as_string().len(),
        };
        if payload_size > limits.max_payload_bytes {
            return Err(CommandError::LimitExceeded("payload"));
        }
        Ok(Self {
            id,
            actor,
            timestamp,
            causal: normalized_causal,
            affected_domains,
            payload,
        })
    }

    /// Returns the deterministic total-order key used for concurrent replay.
    #[must_use]
    pub fn order_key(&self) -> (&HlcTimestamp, &ActorId, u64) {
        (&self.timestamp, &self.actor, self.id.sequence)
    }
}

fn value_size(value: &Value) -> usize {
    match value {
        Value::Null | Value::Bool(_) | Value::Int(_) | Value::UInt(_) | Value::FloatBits(_) => 8,
        Value::Text(text) => text.len(),
        Value::Bytes(bytes) => bytes.len(),
    }
}
