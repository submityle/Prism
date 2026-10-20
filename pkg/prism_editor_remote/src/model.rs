use std::collections::BTreeMap;
use std::fmt;

/// Protocol version understood by a synchronization endpoint.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ProtocolVersion {
    /// Major version changes are incompatible.
    pub major: u16,
    /// Minor version changes are additive when capabilities permit them.
    pub minor: u16,
}

impl ProtocolVersion {
    /// The first version of this protocol.
    pub const CURRENT: Self = Self { major: 1, minor: 0 };
}

/// Correlates one request with exactly one response.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct RequestId(u64);

impl RequestId {
    /// Creates an ID from its wire representation.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the wire representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Identifies a runtime document, world, or other synchronization target.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct TargetId(u128);

impl TargetId {
    /// Creates an ID from its stable numeric representation.
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    /// Returns the stable numeric representation.
    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Identifies one runtime entity without imposing an ECS implementation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct EntityId(u128);

impl EntityId {
    /// Creates an entity ID from its stable numeric representation.
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    /// Returns the stable numeric representation.
    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Monotonically increasing revision of a synchronization target.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct Revision(u64);

impl Revision {
    /// Revision zero is the initial empty history.
    pub const ZERO: Self = Self(0);

    /// Creates a revision from its wire representation.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the wire representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Returns the next revision, saturating at the maximum wire value.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

/// Stable, transport-neutral component type identifier.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ComponentId(String);

impl ComponentId {
    /// Creates a component ID. Empty IDs are rejected because they cannot be
    /// diagnosed reliably at an adapter boundary.
    pub fn new(value: impl Into<String>) -> Result<Self, SyncError> {
        let value = value.into();
        if value.is_empty() {
            return Err(SyncError::InvalidRequest {
                code: ErrorCode::InvalidIdentifier,
                detail: String::from("component identifiers must not be empty"),
            });
        }
        Ok(Self(value))
    }

    /// Creates an ID for trusted static data.
    #[must_use]
    pub fn from_static(value: &'static str) -> Self {
        assert!(
            !value.is_empty(),
            "static component identifiers must not be empty"
        );
        Self(String::from(value))
    }

    /// Returns the canonical component name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComponentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A typed value used by snapshots and changes without JSON semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Value {
    /// No value.
    Null,
    /// A boolean.
    Bool(bool),
    /// A signed integer.
    Signed(i64),
    /// An unsigned integer.
    Unsigned(u64),
    /// An IEEE-754 value stored as exact bits for deterministic equality.
    Float(u64),
    /// A UTF-8 string.
    String(String),
    /// Opaque bytes.
    Bytes(Vec<u8>),
    /// An ordered sequence.
    Array(Vec<Self>),
    /// An ordered string-keyed object.
    Object(BTreeMap<String, Self>),
}

impl Value {
    /// Stores an `f64` while preserving its exact bit pattern.
    #[must_use]
    pub fn float(value: f64) -> Self {
        Self::Float(value.to_bits())
    }

    /// Converts an exact float value back to `f64`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }

    /// Creates an object from key/value pairs, rejecting duplicate keys.
    pub fn object<I, K>(entries: I) -> Result<Self, SyncError>
    where
        I: IntoIterator<Item = (K, Self)>,
        K: Into<String>,
    {
        let mut object = BTreeMap::new();
        for (key, value) in entries {
            let key = key.into();
            if object.insert(key.clone(), value).is_some() {
                return Err(SyncError::InvalidRequest {
                    code: ErrorCode::DuplicateKey,
                    detail: format!("object key `{key}` appears more than once"),
                });
            }
        }
        Ok(Self::Object(object))
    }
}

/// One segment in a property path.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum PropertySegment {
    /// A named field or object key.
    Field(String),
    /// An array index.
    Index(u32),
    /// An explicitly keyed object access.
    Key(String),
}

/// A canonical path below a component value.
#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct PropertyPath(Vec<PropertySegment>);

impl PropertyPath {
    /// Returns the root path, which addresses the complete component value.
    #[must_use]
    pub const fn root() -> Self {
        Self(Vec::new())
    }

    /// Builds a path and rejects empty field/key segments.
    pub fn new(segments: Vec<PropertySegment>) -> Result<Self, SyncError> {
        for segment in &segments {
            let text = match segment {
                PropertySegment::Field(text) | PropertySegment::Key(text) => Some(text),
                PropertySegment::Index(_) => None,
            };
            if text.is_some_and(String::is_empty) {
                return Err(SyncError::InvalidRequest {
                    code: ErrorCode::InvalidPropertyPath,
                    detail: String::from("property path fields and keys must not be empty"),
                });
            }
        }
        Ok(Self(segments))
    }

    /// Returns the path segments.
    #[must_use]
    pub fn segments(&self) -> &[PropertySegment] {
        &self.0
    }

    /// Returns true when the path addresses the component root.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns a stable human-readable path form.
    #[must_use]
    pub fn as_string(&self) -> String {
        let mut result = String::new();
        for segment in &self.0 {
            match segment {
                PropertySegment::Field(name) => {
                    result.push('.');
                    result.push_str(name);
                }
                PropertySegment::Index(index) => {
                    result.push('[');
                    result.push_str(&index.to_string());
                    result.push(']');
                }
                PropertySegment::Key(key) => {
                    result.push_str("[\"");
                    result.push_str(key);
                    result.push_str("\"]");
                }
            }
        }
        if result.is_empty() {
            String::from("<root>")
        } else {
            result
        }
    }
}

/// Capabilities advertised by an endpoint.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum Capability {
    /// The endpoint can return complete snapshots.
    SnapshotRead,
    /// The endpoint can return changes after a revision.
    IncrementalRead,
    /// The endpoint can apply component-level changes.
    ComponentWrite,
    /// The endpoint can apply property-level changes.
    PropertyWrite,
    /// The endpoint checks a caller-provided base revision.
    RevisionChecks,
    /// The endpoint and adapter support the deterministic binary codec.
    DeterministicCodec,
}

impl Capability {
    const fn bit(self) -> u32 {
        match self {
            Self::SnapshotRead => 1 << 0,
            Self::IncrementalRead => 1 << 1,
            Self::ComponentWrite => 1 << 2,
            Self::PropertyWrite => 1 << 3,
            Self::RevisionChecks => 1 << 4,
            Self::DeterministicCodec => 1 << 5,
        }
    }
}

/// A compact set of endpoint capabilities.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct EndpointCapabilities(u32);

impl EndpointCapabilities {
    /// Creates an empty capability set.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Creates a set containing the supplied capabilities.
    #[must_use]
    pub fn from_capabilities<I: IntoIterator<Item = Capability>>(capabilities: I) -> Self {
        let mut result = Self::empty();
        for capability in capabilities {
            result.insert(capability);
        }
        result
    }

    /// Adds one capability.
    pub const fn insert(&mut self, capability: Capability) {
        self.0 |= capability.bit();
    }

    /// Returns whether a capability is present.
    #[must_use]
    pub const fn contains(self, capability: Capability) -> bool {
        self.0 & capability.bit() != 0
    }

    /// Returns the stable wire bitset.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Creates a set from a wire bitset, rejecting unknown future bits.
    pub fn from_bits(bits: u32) -> Result<Self, SyncError> {
        let known =
            Capability::DeterministicCodec.bit() | (Capability::DeterministicCodec.bit() - 1);
        if bits & !known != 0 {
            return Err(SyncError::InvalidRequest {
                code: ErrorCode::InvalidCapabilityBits,
                detail: format!("unknown capability bits: 0x{:08x}", bits & !known),
            });
        }
        Ok(Self(bits))
    }
}

/// Endpoint metadata exchanged before synchronization begins.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EndpointDescriptor {
    /// Protocol version implemented by the endpoint.
    pub protocol: ProtocolVersion,
    /// Human-readable endpoint name.
    pub name: String,
    /// Behavior supported by the endpoint.
    pub capabilities: EndpointCapabilities,
    /// Targets currently available from the endpoint.
    pub targets: Vec<TargetId>,
}

impl EndpointDescriptor {
    /// Creates a descriptor and canonicalizes target ordering.
    #[must_use]
    pub fn new(name: impl Into<String>, capabilities: EndpointCapabilities) -> Self {
        Self {
            protocol: ProtocolVersion::CURRENT,
            name: name.into(),
            capabilities,
            targets: Vec::new(),
        }
    }

    /// Adds a target if it is not already advertised.
    pub fn add_target(&mut self, target: TargetId) {
        if !self.targets.contains(&target) {
            self.targets.push(target);
            self.targets.sort_unstable();
        }
    }
}

/// One entity and its component values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EntityRecord {
    /// Component values keyed by stable component ID.
    pub components: BTreeMap<ComponentId, Value>,
}

impl EntityRecord {
    /// Creates an empty entity record.
    #[must_use]
    pub fn new() -> Self {
        Self {
            components: BTreeMap::new(),
        }
    }
}

impl Default for EntityRecord {
    fn default() -> Self {
        Self::new()
    }
}

/// A complete synchronization snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldSnapshot {
    /// Target represented by this snapshot.
    pub target: TargetId,
    /// Revision represented by this snapshot.
    pub revision: Revision,
    /// Entities keyed in deterministic order.
    pub entities: BTreeMap<EntityId, EntityRecord>,
    /// Optional target metadata, also deterministic.
    pub metadata: BTreeMap<String, Value>,
}

impl WorldSnapshot {
    /// Creates an empty target at revision zero.
    #[must_use]
    pub fn empty(target: TargetId) -> Self {
        Self {
            target,
            revision: Revision::ZERO,
            entities: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }
}

/// A component/property mutation applied in order.
#[allow(missing_docs)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Change {
    /// Creates an empty entity.
    CreateEntity { entity: EntityId },
    /// Deletes an entity and all of its components.
    DeleteEntity { entity: EntityId },
    /// Inserts or replaces a complete component value.
    SetComponent {
        entity: EntityId,
        component: ComponentId,
        value: Value,
    },
    /// Removes an existing component.
    RemoveComponent {
        entity: EntityId,
        component: ComponentId,
    },
    /// Inserts or replaces a nested property. The root path replaces the
    /// complete component value.
    SetProperty {
        entity: EntityId,
        component: ComponentId,
        path: PropertyPath,
        value: Value,
    },
    /// Removes a nested object property. The root path is rejected; use
    /// `RemoveComponent` for that operation.
    RemoveProperty {
        entity: EntityId,
        component: ComponentId,
        path: PropertyPath,
    },
}

/// A validated ordered set of mutations between two revisions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChangeSet {
    /// Revision the sender read before producing these changes.
    pub base_revision: Revision,
    /// Revision after these changes are applied.
    pub target_revision: Revision,
    /// Ordered changes; order is semantically significant.
    pub changes: Vec<Change>,
}

impl ChangeSet {
    /// Creates a change set after checking revision monotonicity.
    pub fn new(
        base_revision: Revision,
        target_revision: Revision,
        changes: Vec<Change>,
    ) -> Result<Self, SyncError> {
        if target_revision < base_revision {
            return Err(SyncError::InvalidRequest {
                code: ErrorCode::InvalidRevisionRange,
                detail: format!(
                    "target revision {} precedes base revision {}",
                    target_revision.raw(),
                    base_revision.raw()
                ),
            });
        }
        if changes.is_empty() && target_revision != base_revision {
            return Err(SyncError::InvalidRequest {
                code: ErrorCode::InvalidRevisionRange,
                detail: String::from("a non-empty revision range requires at least one change"),
            });
        }
        if !changes.is_empty() && target_revision == base_revision {
            return Err(SyncError::InvalidRequest {
                code: ErrorCode::InvalidRevisionRange,
                detail: String::from("changes must advance the target revision"),
            });
        }
        Ok(Self {
            base_revision,
            target_revision,
            changes,
        })
    }

    /// Creates an empty change set at one revision.
    #[must_use]
    pub fn empty(revision: Revision) -> Self {
        Self {
            base_revision: revision,
            target_revision: revision,
            changes: Vec::new(),
        }
    }
}

/// Requests accepted by an endpoint.
#[allow(missing_docs)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Request {
    /// Requests endpoint metadata and capabilities.
    Describe,
    /// Requests a complete snapshot.
    ReadSnapshot { target: TargetId },
    /// Requests all retained changes after a revision.
    ReadChanges {
        target: TargetId,
        from_revision: Revision,
    },
    /// Applies ordered changes against an exact base revision.
    ApplyChanges {
        target: TargetId,
        base_revision: Revision,
        changes: Vec<Change>,
    },
    /// A transport-neutral liveness check.
    Ping,
}

/// A request plus its correlation ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestEnvelope {
    /// Correlation ID echoed by the response.
    pub request_id: RequestId,
    /// Request payload.
    pub request: Request,
}

/// Successful response payloads.
#[allow(missing_docs)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResponsePayload {
    /// Endpoint metadata.
    Descriptor(EndpointDescriptor),
    /// Complete target state.
    Snapshot(WorldSnapshot),
    /// Incremental target changes.
    Changes(ChangeSet),
    /// Result of an accepted mutation.
    Commit {
        target: TargetId,
        revision: Revision,
        changes: Vec<Change>,
    },
    /// Response to [`Request::Ping`].
    Pong,
}

/// A response is either an explicit success payload or an explicit protocol
/// error; it is never represented by an ambiguous empty value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Response {
    /// Successful operation.
    Success(ResponsePayload),
    /// Failed operation with a stable error category.
    Error(SyncError),
}

/// A response plus the request ID it answers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponseEnvelope {
    /// Correlation ID copied from the request.
    pub request_id: RequestId,
    /// Response payload or explicit error.
    pub response: Response,
}

/// Stable error category used by protocol responses and codec diagnostics.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum ErrorCode {
    /// Request shape or value is invalid.
    InvalidRequest,
    /// An identifier violates its contract.
    InvalidIdentifier,
    /// A property path is invalid.
    InvalidPropertyPath,
    /// An object contains a duplicate key.
    DuplicateKey,
    /// Capability bitset contains an unknown bit.
    InvalidCapabilityBits,
    /// Requested operation is not advertised.
    UnsupportedCapability,
    /// Requested target is not available.
    UnknownTarget,
    /// Caller base revision differs from endpoint revision.
    RevisionConflict,
    /// Requested history is no longer retained.
    RevisionUnavailable,
    /// Entity does not exist.
    MissingEntity,
    /// Entity already exists.
    DuplicateEntity,
    /// Component does not exist.
    MissingComponent,
    /// Property does not exist.
    MissingProperty,
    /// Operation cannot be performed on this value shape.
    InvalidValueShape,
    /// Revision range is malformed.
    InvalidRevisionRange,
    /// Replica received changes for another target.
    TargetMismatch,
    /// Replica received a non-contiguous revision.
    RevisionGap,
}

/// Explicit protocol/storage error state.
#[allow(missing_docs)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SyncError {
    /// Invalid request with a stable code and diagnostic detail.
    InvalidRequest { code: ErrorCode, detail: String },
    /// A request requires a capability that is not advertised.
    UnsupportedCapability { capability: Capability },
    /// The target is not known by the endpoint.
    UnknownTarget { target: TargetId },
    /// An optimistic write used a stale revision.
    RevisionConflict {
        target: TargetId,
        expected: Revision,
        actual: Revision,
    },
    /// The endpoint cannot provide the requested historical range.
    RevisionUnavailable {
        target: TargetId,
        requested: Revision,
        oldest_available: Revision,
    },
    /// A referenced entity is absent.
    MissingEntity { entity: EntityId },
    /// An entity creation would overwrite an existing entity.
    DuplicateEntity { entity: EntityId },
    /// A referenced component is absent.
    MissingComponent {
        entity: EntityId,
        component: ComponentId,
    },
    /// A referenced property is absent.
    MissingProperty { path: PropertyPath },
    /// A nested operation does not match the value shape.
    InvalidValueShape { path: PropertyPath, detail: String },
    /// A request targets another replica or endpoint.
    TargetMismatch {
        expected: TargetId,
        actual: TargetId,
    },
    /// A replica received a non-contiguous change set.
    RevisionGap {
        expected: Revision,
        actual: Revision,
    },
}

impl SyncError {
    /// Returns the stable error code.
    #[must_use]
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::InvalidRequest { code, .. } => *code,
            Self::UnsupportedCapability { .. } => ErrorCode::UnsupportedCapability,
            Self::UnknownTarget { .. } => ErrorCode::UnknownTarget,
            Self::RevisionConflict { .. } => ErrorCode::RevisionConflict,
            Self::RevisionUnavailable { .. } => ErrorCode::RevisionUnavailable,
            Self::MissingEntity { .. } => ErrorCode::MissingEntity,
            Self::DuplicateEntity { .. } => ErrorCode::DuplicateEntity,
            Self::MissingComponent { .. } => ErrorCode::MissingComponent,
            Self::MissingProperty { .. } => ErrorCode::MissingProperty,
            Self::InvalidValueShape { .. } => ErrorCode::InvalidValueShape,
            Self::TargetMismatch { .. } => ErrorCode::TargetMismatch,
            Self::RevisionGap { .. } => ErrorCode::RevisionGap,
        }
    }
}

impl fmt::Display for SyncError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest { detail, .. } => write!(formatter, "invalid request: {detail}"),
            Self::UnsupportedCapability { capability } => {
                write!(formatter, "unsupported capability: {capability:?}")
            }
            Self::UnknownTarget { target } => write!(formatter, "unknown target {}", target.raw()),
            Self::RevisionConflict {
                target,
                expected,
                actual,
            } => write!(
                formatter,
                "revision conflict for target {}: expected {}, actual {}",
                target.raw(),
                expected.raw(),
                actual.raw()
            ),
            Self::RevisionUnavailable {
                target,
                requested,
                oldest_available,
            } => write!(
                formatter,
                "revision {} is unavailable for target {}; oldest is {}",
                requested.raw(),
                target.raw(),
                oldest_available.raw()
            ),
            Self::MissingEntity { entity } => write!(formatter, "missing entity {}", entity.raw()),
            Self::DuplicateEntity { entity } => {
                write!(formatter, "duplicate entity {}", entity.raw())
            }
            Self::MissingComponent { entity, component } => write!(
                formatter,
                "missing component {component} on entity {}",
                entity.raw()
            ),
            Self::MissingProperty { path } => {
                write!(formatter, "missing property {}", path.as_string())
            }
            Self::InvalidValueShape { path, detail } => {
                write!(formatter, "invalid value at {}: {detail}", path.as_string())
            }
            Self::TargetMismatch { expected, actual } => write!(
                formatter,
                "target mismatch: expected {}, received {}",
                expected.raw(),
                actual.raw()
            ),
            Self::RevisionGap { expected, actual } => write!(
                formatter,
                "revision gap: expected {}, received {}",
                expected.raw(),
                actual.raw()
            ),
        }
    }
}

impl std::error::Error for SyncError {}

pub(crate) fn apply_change(snapshot: &mut WorldSnapshot, change: &Change) -> Result<(), SyncError> {
    match change {
        Change::CreateEntity { entity } => {
            if snapshot
                .entities
                .insert(*entity, EntityRecord::new())
                .is_some()
            {
                return Err(SyncError::DuplicateEntity { entity: *entity });
            }
        }
        Change::DeleteEntity { entity } => {
            if snapshot.entities.remove(entity).is_none() {
                return Err(SyncError::MissingEntity { entity: *entity });
            }
        }
        Change::SetComponent {
            entity,
            component,
            value,
        } => {
            let record = snapshot
                .entities
                .get_mut(entity)
                .ok_or(SyncError::MissingEntity { entity: *entity })?;
            record.components.insert(component.clone(), value.clone());
        }
        Change::RemoveComponent { entity, component } => {
            let record = snapshot
                .entities
                .get_mut(entity)
                .ok_or(SyncError::MissingEntity { entity: *entity })?;
            if record.components.remove(component).is_none() {
                return Err(SyncError::MissingComponent {
                    entity: *entity,
                    component: component.clone(),
                });
            }
        }
        Change::SetProperty {
            entity,
            component,
            path,
            value,
        } => {
            let record = snapshot
                .entities
                .get_mut(entity)
                .ok_or(SyncError::MissingEntity { entity: *entity })?;
            let root = record.components.get_mut(component).ok_or_else(|| {
                SyncError::MissingComponent {
                    entity: *entity,
                    component: component.clone(),
                }
            })?;
            set_property(root, path, value.clone())?;
        }
        Change::RemoveProperty {
            entity,
            component,
            path,
        } => {
            let record = snapshot
                .entities
                .get_mut(entity)
                .ok_or(SyncError::MissingEntity { entity: *entity })?;
            let root = record.components.get_mut(component).ok_or_else(|| {
                SyncError::MissingComponent {
                    entity: *entity,
                    component: component.clone(),
                }
            })?;
            remove_property(root, path)?;
        }
    }
    Ok(())
}

pub(crate) fn apply_changes(
    snapshot: &mut WorldSnapshot,
    changes: &[Change],
) -> Result<(), SyncError> {
    for change in changes {
        apply_change(snapshot, change)?;
    }
    Ok(())
}

fn set_property(root: &mut Value, path: &PropertyPath, value: Value) -> Result<(), SyncError> {
    if path.is_root() {
        *root = value;
        return Ok(());
    }
    let (last, parent) = path
        .segments()
        .split_last()
        .expect("non-root path has a final segment");
    let parent = descend_mut(root, parent, path)?;
    match (parent, last) {
        (Value::Object(object), PropertySegment::Field(name) | PropertySegment::Key(name)) => {
            object.insert(name.clone(), value);
            Ok(())
        }
        (Value::Array(values), PropertySegment::Index(index)) => {
            let slot = values
                .get_mut(*index as usize)
                .ok_or_else(|| SyncError::MissingProperty { path: path.clone() })?;
            *slot = value;
            Ok(())
        }
        (_, segment) => Err(SyncError::InvalidValueShape {
            path: path.clone(),
            detail: format!("cannot use {segment:?} on this parent value"),
        }),
    }
}

fn remove_property(root: &mut Value, path: &PropertyPath) -> Result<(), SyncError> {
    if path.is_root() {
        return Err(SyncError::InvalidValueShape {
            path: path.clone(),
            detail: String::from("the component root cannot be removed as a property"),
        });
    }
    let (last, parent) = path
        .segments()
        .split_last()
        .expect("non-root path has a final segment");
    let parent = descend_mut(root, parent, path)?;
    match (parent, last) {
        (Value::Object(object), PropertySegment::Field(name) | PropertySegment::Key(name)) => {
            if object.remove(name).is_none() {
                return Err(SyncError::MissingProperty { path: path.clone() });
            }
            Ok(())
        }
        (Value::Array(_), PropertySegment::Index(_)) => Err(SyncError::InvalidValueShape {
            path: path.clone(),
            detail: String::from("removing array elements is not a stable property operation"),
        }),
        (_, segment) => Err(SyncError::InvalidValueShape {
            path: path.clone(),
            detail: format!("cannot use {segment:?} on this parent value"),
        }),
    }
}

fn descend_mut<'a>(
    mut value: &'a mut Value,
    segments: &[PropertySegment],
    full_path: &PropertyPath,
) -> Result<&'a mut Value, SyncError> {
    for segment in segments {
        value = match (value, segment) {
            (Value::Object(object), PropertySegment::Field(name) | PropertySegment::Key(name)) => {
                object
                    .get_mut(name)
                    .ok_or_else(|| SyncError::MissingProperty {
                        path: full_path.clone(),
                    })?
            }
            (Value::Array(values), PropertySegment::Index(index)) => values
                .get_mut(*index as usize)
                .ok_or_else(|| SyncError::MissingProperty {
                    path: full_path.clone(),
                })?,
            (_, segment) => {
                return Err(SyncError::InvalidValueShape {
                    path: full_path.clone(),
                    detail: format!("cannot descend through {segment:?}"),
                });
            }
        };
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_and_values_have_deterministic_forms() {
        let path = PropertyPath::new(vec![
            PropertySegment::Field(String::from("transform")),
            PropertySegment::Key(String::from("x")),
            PropertySegment::Index(2),
        ])
        .expect("path");
        assert_eq!(path.as_string(), ".transform[\"x\"][2]");
        assert_eq!(Value::float(-0.0).as_f64(), Some(-0.0));
    }

    #[test]
    fn nested_properties_are_applied_without_partial_failure() {
        let target = TargetId::from_raw(1);
        let entity = EntityId::from_raw(2);
        let component = ComponentId::from_static("Transform");
        let mut snapshot = WorldSnapshot::empty(target);
        let mut record = EntityRecord::new();
        record.components.insert(
            component.clone(),
            Value::object([(
                "position",
                Value::object([("x", Value::Signed(1))]).expect("object"),
            )])
            .expect("object"),
        );
        snapshot.entities.insert(entity, record);
        let path = PropertyPath::new(vec![
            PropertySegment::Field(String::from("position")),
            PropertySegment::Field(String::from("x")),
        ])
        .expect("path");
        apply_change(
            &mut snapshot,
            &Change::SetProperty {
                entity,
                component: component.clone(),
                path: path.clone(),
                value: Value::Signed(5),
            },
        )
        .expect("set");
        assert_eq!(
            snapshot.entities[&entity].components[&component],
            Value::object([(
                "position",
                Value::object([("x", Value::Signed(5))]).expect("object")
            ),])
            .expect("object")
        );
        assert_eq!(
            apply_change(
                &mut snapshot,
                &Change::RemoveProperty {
                    entity,
                    component,
                    path: PropertyPath::root(),
                },
            )
            .expect_err("root removal must fail")
            .code(),
            ErrorCode::InvalidValueShape
        );
    }
}
