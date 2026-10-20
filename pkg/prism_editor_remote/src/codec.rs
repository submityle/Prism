use crate::model::{
    Capability, Change, ChangeSet, ComponentId, EndpointCapabilities, EndpointDescriptor, EntityId,
    EntityRecord, ErrorCode, PropertyPath, PropertySegment, ProtocolVersion, Request,
    RequestEnvelope, RequestId, Response, ResponseEnvelope, ResponsePayload, Revision, SyncError,
    TargetId, Value, WorldSnapshot,
};
use std::collections::BTreeMap;
use std::fmt;

const MAGIC: [u8; 4] = *b"PER1";
const REQUEST_DESCRIBE: u8 = 0;
const REQUEST_SNAPSHOT: u8 = 1;
const REQUEST_CHANGES: u8 = 2;
const REQUEST_APPLY: u8 = 3;
const REQUEST_PING: u8 = 4;
const RESPONSE_SUCCESS: u8 = 0;
const RESPONSE_ERROR: u8 = 1;

/// Allocation limits used while decoding untrusted frames.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodecLimits {
    /// Maximum complete frame size.
    pub max_frame_bytes: usize,
    /// Maximum UTF-8 string size.
    pub max_string_bytes: usize,
    /// Maximum opaque byte-array size.
    pub max_bytes: usize,
    /// Maximum number of entries in one array/map/list.
    pub max_collection_items: usize,
    /// Maximum nested value depth.
    pub max_value_depth: usize,
}

impl Default for CodecLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 16 * 1024 * 1024,
            max_string_bytes: 1024 * 1024,
            max_bytes: 4 * 1024 * 1024,
            max_collection_items: 100_000,
            max_value_depth: 64,
        }
    }
}

/// Failure while encoding or decoding a deterministic frame.
#[allow(missing_docs)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodecError {
    /// The input ended before a complete field was available.
    Truncated,
    /// The four-byte frame marker is not recognized.
    InvalidMagic,
    /// A tag does not identify a known protocol variant.
    InvalidTag { kind: &'static str, tag: u8 },
    /// A length or count is not representable by the host or configured limit.
    LengthOverflow { kind: &'static str },
    /// The frame exceeds one configured allocation limit.
    LimitExceeded { kind: &'static str, limit: usize },
    /// A string is not valid UTF-8.
    InvalidUtf8,
    /// Bytes remain after a complete message was decoded.
    TrailingBytes { count: usize },
    /// A decoded value violates a model invariant.
    InvalidModel(String),
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => formatter.write_str("truncated synchronization frame"),
            Self::InvalidMagic => formatter.write_str("invalid synchronization frame magic"),
            Self::InvalidTag { kind, tag } => write!(formatter, "invalid {kind} tag {tag}"),
            Self::LengthOverflow { kind } => write!(formatter, "{kind} length overflow"),
            Self::LimitExceeded { kind, limit } => {
                write!(formatter, "{kind} exceeds codec limit {limit}")
            }
            Self::InvalidUtf8 => formatter.write_str("invalid UTF-8 in synchronization frame"),
            Self::TrailingBytes { count } => write!(formatter, "{count} trailing frame bytes"),
            Self::InvalidModel(detail) => {
                write!(formatter, "invalid synchronization model: {detail}")
            }
        }
    }
}

impl std::error::Error for CodecError {}

/// Encodes a request using the default limits.
pub fn encode_request(request: &RequestEnvelope) -> Result<Vec<u8>, CodecError> {
    encode_request_with_limits(request, CodecLimits::default())
}

/// Encodes a request with explicit output limits.
pub fn encode_request_with_limits(
    request: &RequestEnvelope,
    limits: CodecLimits,
) -> Result<Vec<u8>, CodecError> {
    let mut writer = Writer::new(limits);
    writer.put_bytes_raw(&MAGIC);
    writer.put_u8(request_tag(&request.request));
    writer.put_u64(request.request_id.raw());
    encode_request_body(&mut writer, &request.request)?;
    writer.finish()
}

/// Decodes a request using the default limits.
pub fn decode_request(bytes: &[u8]) -> Result<RequestEnvelope, CodecError> {
    decode_request_with_limits(bytes, CodecLimits::default())
}

/// Decodes a request with explicit allocation limits.
pub fn decode_request_with_limits(
    bytes: &[u8],
    limits: CodecLimits,
) -> Result<RequestEnvelope, CodecError> {
    let mut reader = Reader::new(bytes, limits)?;
    reader.read_magic()?;
    let tag = reader.read_u8()?;
    let request_id = RequestId::from_raw(reader.read_u64()?);
    let request = decode_request_body(&mut reader, tag)?;
    reader.finish()?;
    Ok(RequestEnvelope {
        request_id,
        request,
    })
}

/// Encodes a response using the default limits.
pub fn encode_response(response: &ResponseEnvelope) -> Result<Vec<u8>, CodecError> {
    encode_response_with_limits(response, CodecLimits::default())
}

/// Encodes a response with explicit output limits.
pub fn encode_response_with_limits(
    response: &ResponseEnvelope,
    limits: CodecLimits,
) -> Result<Vec<u8>, CodecError> {
    let mut writer = Writer::new(limits);
    writer.put_bytes_raw(&MAGIC);
    writer.put_u64(response.request_id.raw());
    match &response.response {
        Response::Success(payload) => {
            writer.put_u8(RESPONSE_SUCCESS);
            encode_payload(&mut writer, payload)?;
        }
        Response::Error(error) => {
            writer.put_u8(RESPONSE_ERROR);
            encode_error(&mut writer, error)?;
        }
    }
    writer.finish()
}

/// Decodes a response using the default limits.
pub fn decode_response(bytes: &[u8]) -> Result<ResponseEnvelope, CodecError> {
    decode_response_with_limits(bytes, CodecLimits::default())
}

/// Decodes a response with explicit allocation limits.
pub fn decode_response_with_limits(
    bytes: &[u8],
    limits: CodecLimits,
) -> Result<ResponseEnvelope, CodecError> {
    let mut reader = Reader::new(bytes, limits)?;
    reader.read_magic()?;
    let request_id = RequestId::from_raw(reader.read_u64()?);
    let status = reader.read_u8()?;
    let response = match status {
        RESPONSE_SUCCESS => Response::Success(decode_payload(&mut reader)?),
        RESPONSE_ERROR => Response::Error(decode_error(&mut reader)?),
        tag => {
            return Err(CodecError::InvalidTag {
                kind: "response status",
                tag,
            });
        }
    };
    reader.finish()?;
    Ok(ResponseEnvelope {
        request_id,
        response,
    })
}

fn request_tag(request: &Request) -> u8 {
    match request {
        Request::Describe => REQUEST_DESCRIBE,
        Request::ReadSnapshot { .. } => REQUEST_SNAPSHOT,
        Request::ReadChanges { .. } => REQUEST_CHANGES,
        Request::ApplyChanges { .. } => REQUEST_APPLY,
        Request::Ping => REQUEST_PING,
    }
}

fn encode_request_body(writer: &mut Writer, request: &Request) -> Result<(), CodecError> {
    match request {
        Request::Describe | Request::Ping => Ok(()),
        Request::ReadSnapshot { target } => {
            writer.put_u128(target.raw());
            Ok(())
        }
        Request::ReadChanges {
            target,
            from_revision,
        } => {
            writer.put_u128(target.raw());
            writer.put_u64(from_revision.raw());
            Ok(())
        }
        Request::ApplyChanges {
            target,
            base_revision,
            changes,
        } => {
            writer.put_u128(target.raw());
            writer.put_u64(base_revision.raw());
            writer.put_count(changes.len(), "change list")?;
            for change in changes {
                encode_change(writer, change)?;
            }
            Ok(())
        }
    }
}

fn decode_request_body(reader: &mut Reader<'_>, tag: u8) -> Result<Request, CodecError> {
    match tag {
        REQUEST_DESCRIBE => Ok(Request::Describe),
        REQUEST_SNAPSHOT => Ok(Request::ReadSnapshot {
            target: TargetId::from_raw(reader.read_u128()?),
        }),
        REQUEST_CHANGES => Ok(Request::ReadChanges {
            target: TargetId::from_raw(reader.read_u128()?),
            from_revision: Revision::from_raw(reader.read_u64()?),
        }),
        REQUEST_APPLY => {
            let target = TargetId::from_raw(reader.read_u128()?);
            let base_revision = Revision::from_raw(reader.read_u64()?);
            let count = reader.read_count("change list")?;
            let mut changes = Vec::with_capacity(count);
            for _ in 0..count {
                changes.push(decode_change(reader)?);
            }
            Ok(Request::ApplyChanges {
                target,
                base_revision,
                changes,
            })
        }
        REQUEST_PING => Ok(Request::Ping),
        tag => Err(CodecError::InvalidTag {
            kind: "request",
            tag,
        }),
    }
}

fn encode_payload(writer: &mut Writer, payload: &ResponsePayload) -> Result<(), CodecError> {
    match payload {
        ResponsePayload::Descriptor(descriptor) => {
            writer.put_u8(0);
            encode_descriptor(writer, descriptor)
        }
        ResponsePayload::Snapshot(snapshot) => {
            writer.put_u8(1);
            encode_snapshot(writer, snapshot)
        }
        ResponsePayload::Changes(changes) => {
            writer.put_u8(2);
            encode_change_set(writer, changes)
        }
        ResponsePayload::Commit {
            target,
            revision,
            changes,
        } => {
            writer.put_u8(3);
            writer.put_u128(target.raw());
            writer.put_u64(revision.raw());
            writer.put_count(changes.len(), "commit change list")?;
            for change in changes {
                encode_change(writer, change)?;
            }
            Ok(())
        }
        ResponsePayload::Pong => {
            writer.put_u8(4);
            Ok(())
        }
    }
}

fn decode_payload(reader: &mut Reader<'_>) -> Result<ResponsePayload, CodecError> {
    match reader.read_u8()? {
        0 => Ok(ResponsePayload::Descriptor(decode_descriptor(reader)?)),
        1 => Ok(ResponsePayload::Snapshot(decode_snapshot(reader)?)),
        2 => Ok(ResponsePayload::Changes(decode_change_set(reader)?)),
        3 => {
            let target = TargetId::from_raw(reader.read_u128()?);
            let revision = Revision::from_raw(reader.read_u64()?);
            let count = reader.read_count("commit change list")?;
            let mut changes = Vec::with_capacity(count);
            for _ in 0..count {
                changes.push(decode_change(reader)?);
            }
            Ok(ResponsePayload::Commit {
                target,
                revision,
                changes,
            })
        }
        4 => Ok(ResponsePayload::Pong),
        tag => Err(CodecError::InvalidTag {
            kind: "response payload",
            tag,
        }),
    }
}

fn encode_descriptor(
    writer: &mut Writer,
    descriptor: &EndpointDescriptor,
) -> Result<(), CodecError> {
    writer.put_u16(descriptor.protocol.major);
    writer.put_u16(descriptor.protocol.minor);
    writer.put_string(&descriptor.name)?;
    writer.put_u32(descriptor.capabilities.bits());
    writer.put_count(descriptor.targets.len(), "target list")?;
    for target in &descriptor.targets {
        writer.put_u128(target.raw());
    }
    Ok(())
}

fn decode_descriptor(reader: &mut Reader<'_>) -> Result<EndpointDescriptor, CodecError> {
    let protocol = ProtocolVersion {
        major: reader.read_u16()?,
        minor: reader.read_u16()?,
    };
    let name = reader.read_string()?;
    let capabilities =
        EndpointCapabilities::from_bits(reader.read_u32()?).map_err(sync_to_codec)?;
    let count = reader.read_count("target list")?;
    let mut targets = Vec::with_capacity(count);
    for _ in 0..count {
        targets.push(TargetId::from_raw(reader.read_u128()?));
    }
    targets.sort_unstable();
    targets.dedup();
    Ok(EndpointDescriptor {
        protocol,
        name,
        capabilities,
        targets,
    })
}

fn encode_snapshot(writer: &mut Writer, snapshot: &WorldSnapshot) -> Result<(), CodecError> {
    writer.put_u128(snapshot.target.raw());
    writer.put_u64(snapshot.revision.raw());
    writer.put_count(snapshot.entities.len(), "entity map")?;
    for (entity, record) in &snapshot.entities {
        writer.put_u128(entity.raw());
        encode_record(writer, record)?;
    }
    encode_value_map(writer, &snapshot.metadata, 0)
}

fn decode_snapshot(reader: &mut Reader<'_>) -> Result<WorldSnapshot, CodecError> {
    let target = TargetId::from_raw(reader.read_u128()?);
    let revision = Revision::from_raw(reader.read_u64()?);
    let count = reader.read_count("entity map")?;
    let mut entities = BTreeMap::new();
    for _ in 0..count {
        let entity = EntityId::from_raw(reader.read_u128()?);
        let previous = entities.insert(entity, decode_record(reader)?);
        if previous.is_some() {
            return Err(CodecError::InvalidModel(String::from(
                "snapshot contains duplicate entity IDs",
            )));
        }
    }
    let metadata = decode_value_map(reader, 0)?;
    Ok(WorldSnapshot {
        target,
        revision,
        entities,
        metadata,
    })
}

fn encode_record(writer: &mut Writer, record: &EntityRecord) -> Result<(), CodecError> {
    writer.put_count(record.components.len(), "component map")?;
    for (component, value) in &record.components {
        writer.put_string(component.as_str())?;
        encode_value(writer, value, 0)?;
    }
    Ok(())
}

fn decode_record(reader: &mut Reader<'_>) -> Result<EntityRecord, CodecError> {
    let count = reader.read_count("component map")?;
    let mut components = BTreeMap::new();
    for _ in 0..count {
        let component = ComponentId::new(reader.read_string()?).map_err(sync_to_codec)?;
        let previous = components.insert(component, decode_value(reader, 0)?);
        if previous.is_some() {
            return Err(CodecError::InvalidModel(String::from(
                "entity contains duplicate component IDs",
            )));
        }
    }
    Ok(EntityRecord { components })
}

fn encode_change_set(writer: &mut Writer, changes: &ChangeSet) -> Result<(), CodecError> {
    writer.put_u64(changes.base_revision.raw());
    writer.put_u64(changes.target_revision.raw());
    writer.put_count(changes.changes.len(), "change list")?;
    for change in &changes.changes {
        encode_change(writer, change)?;
    }
    Ok(())
}

fn decode_change_set(reader: &mut Reader<'_>) -> Result<ChangeSet, CodecError> {
    let base = Revision::from_raw(reader.read_u64()?);
    let target = Revision::from_raw(reader.read_u64()?);
    let count = reader.read_count("change list")?;
    let mut changes = Vec::with_capacity(count);
    for _ in 0..count {
        changes.push(decode_change(reader)?);
    }
    ChangeSet::new(base, target, changes).map_err(sync_to_codec)
}

fn encode_change(writer: &mut Writer, change: &Change) -> Result<(), CodecError> {
    match change {
        Change::CreateEntity { entity } => {
            writer.put_u8(0);
            writer.put_u128(entity.raw());
        }
        Change::DeleteEntity { entity } => {
            writer.put_u8(1);
            writer.put_u128(entity.raw());
        }
        Change::SetComponent {
            entity,
            component,
            value,
        } => {
            writer.put_u8(2);
            writer.put_u128(entity.raw());
            writer.put_string(component.as_str())?;
            encode_value(writer, value, 0)?;
        }
        Change::RemoveComponent { entity, component } => {
            writer.put_u8(3);
            writer.put_u128(entity.raw());
            writer.put_string(component.as_str())?;
        }
        Change::SetProperty {
            entity,
            component,
            path,
            value,
        } => {
            writer.put_u8(4);
            writer.put_u128(entity.raw());
            writer.put_string(component.as_str())?;
            encode_path(writer, path)?;
            encode_value(writer, value, 0)?;
        }
        Change::RemoveProperty {
            entity,
            component,
            path,
        } => {
            writer.put_u8(5);
            writer.put_u128(entity.raw());
            writer.put_string(component.as_str())?;
            encode_path(writer, path)?;
        }
    }
    Ok(())
}

fn decode_change(reader: &mut Reader<'_>) -> Result<Change, CodecError> {
    let tag = reader.read_u8()?;
    let mut entity =
        || -> Result<EntityId, CodecError> { Ok(EntityId::from_raw(reader.read_u128()?)) };
    match tag {
        0 => Ok(Change::CreateEntity { entity: entity()? }),
        1 => Ok(Change::DeleteEntity { entity: entity()? }),
        2 => Ok(Change::SetComponent {
            entity: entity()?,
            component: ComponentId::new(reader.read_string()?).map_err(sync_to_codec)?,
            value: decode_value(reader, 0)?,
        }),
        3 => Ok(Change::RemoveComponent {
            entity: entity()?,
            component: ComponentId::new(reader.read_string()?).map_err(sync_to_codec)?,
        }),
        4 => Ok(Change::SetProperty {
            entity: entity()?,
            component: ComponentId::new(reader.read_string()?).map_err(sync_to_codec)?,
            path: decode_path(reader)?,
            value: decode_value(reader, 0)?,
        }),
        5 => Ok(Change::RemoveProperty {
            entity: entity()?,
            component: ComponentId::new(reader.read_string()?).map_err(sync_to_codec)?,
            path: decode_path(reader)?,
        }),
        tag => Err(CodecError::InvalidTag {
            kind: "change",
            tag,
        }),
    }
}

fn encode_path(writer: &mut Writer, path: &PropertyPath) -> Result<(), CodecError> {
    writer.put_count(path.segments().len(), "property path")?;
    for segment in path.segments() {
        match segment {
            PropertySegment::Field(name) => {
                writer.put_u8(0);
                writer.put_string(name)?;
            }
            PropertySegment::Index(index) => {
                writer.put_u8(1);
                writer.put_u32(*index);
            }
            PropertySegment::Key(key) => {
                writer.put_u8(2);
                writer.put_string(key)?;
            }
        }
    }
    Ok(())
}

fn decode_path(reader: &mut Reader<'_>) -> Result<PropertyPath, CodecError> {
    let count = reader.read_count("property path")?;
    let mut segments = Vec::with_capacity(count);
    for _ in 0..count {
        segments.push(match reader.read_u8()? {
            0 => PropertySegment::Field(reader.read_string()?),
            1 => PropertySegment::Index(reader.read_u32()?),
            2 => PropertySegment::Key(reader.read_string()?),
            tag => {
                return Err(CodecError::InvalidTag {
                    kind: "property segment",
                    tag,
                });
            }
        });
    }
    PropertyPath::new(segments).map_err(sync_to_codec)
}

fn encode_value(writer: &mut Writer, value: &Value, depth: usize) -> Result<(), CodecError> {
    if depth > writer.limits.max_value_depth {
        return Err(CodecError::LimitExceeded {
            kind: "value depth",
            limit: writer.limits.max_value_depth,
        });
    }
    match value {
        Value::Null => writer.put_u8(0),
        Value::Bool(value) => {
            writer.put_u8(1);
            writer.put_u8(u8::from(*value));
        }
        Value::Signed(value) => {
            writer.put_u8(2);
            writer.put_i64(*value);
        }
        Value::Unsigned(value) => {
            writer.put_u8(3);
            writer.put_u64(*value);
        }
        Value::Float(bits) => {
            writer.put_u8(4);
            writer.put_u64(*bits);
        }
        Value::String(value) => {
            writer.put_u8(5);
            writer.put_string(value)?;
        }
        Value::Bytes(value) => {
            writer.put_u8(6);
            writer.put_bytes(value)?;
        }
        Value::Array(values) => {
            writer.put_u8(7);
            writer.put_count(values.len(), "value array")?;
            for value in values {
                encode_value(writer, value, depth + 1)?;
            }
        }
        Value::Object(object) => {
            writer.put_u8(8);
            encode_value_map(writer, object, depth + 1)?;
        }
    }
    Ok(())
}

fn decode_value(reader: &mut Reader<'_>, depth: usize) -> Result<Value, CodecError> {
    if depth > reader.limits.max_value_depth {
        return Err(CodecError::LimitExceeded {
            kind: "value depth",
            limit: reader.limits.max_value_depth,
        });
    }
    match reader.read_u8()? {
        0 => Ok(Value::Null),
        1 => Ok(Value::Bool(match reader.read_u8()? {
            0 => false,
            1 => true,
            tag => {
                return Err(CodecError::InvalidTag {
                    kind: "boolean",
                    tag,
                });
            }
        })),
        2 => Ok(Value::Signed(reader.read_i64()?)),
        3 => Ok(Value::Unsigned(reader.read_u64()?)),
        4 => Ok(Value::Float(reader.read_u64()?)),
        5 => Ok(Value::String(reader.read_string()?)),
        6 => Ok(Value::Bytes(reader.read_bytes()?)),
        7 => {
            let count = reader.read_count("value array")?;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(decode_value(reader, depth + 1)?);
            }
            Ok(Value::Array(values))
        }
        8 => Ok(Value::Object(decode_value_map(reader, depth + 1)?)),
        tag => Err(CodecError::InvalidTag { kind: "value", tag }),
    }
}

fn encode_value_map(
    writer: &mut Writer,
    object: &BTreeMap<String, Value>,
    depth: usize,
) -> Result<(), CodecError> {
    writer.put_count(object.len(), "value object")?;
    for (key, value) in object {
        writer.put_string(key)?;
        encode_value(writer, value, depth)?;
    }
    Ok(())
}

fn decode_value_map(
    reader: &mut Reader<'_>,
    depth: usize,
) -> Result<BTreeMap<String, Value>, CodecError> {
    let count = reader.read_count("value object")?;
    let mut object = BTreeMap::new();
    for _ in 0..count {
        let key = reader.read_string()?;
        let previous = object.insert(key, decode_value(reader, depth)?);
        if previous.is_some() {
            return Err(CodecError::InvalidModel(String::from(
                "value object contains duplicate keys",
            )));
        }
    }
    Ok(object)
}

fn encode_error(writer: &mut Writer, error: &SyncError) -> Result<(), CodecError> {
    match error {
        SyncError::InvalidRequest { code, detail } => {
            writer.put_u8(0);
            encode_error_code(writer, *code);
            writer.put_string(detail)?;
        }
        SyncError::UnsupportedCapability { capability } => {
            writer.put_u8(1);
            writer.put_u8(capability_tag(*capability));
        }
        SyncError::UnknownTarget { target } => {
            writer.put_u8(2);
            writer.put_u128(target.raw());
        }
        SyncError::RevisionConflict {
            target,
            expected,
            actual,
        } => {
            writer.put_u8(3);
            writer.put_u128(target.raw());
            writer.put_u64(expected.raw());
            writer.put_u64(actual.raw());
        }
        SyncError::RevisionUnavailable {
            target,
            requested,
            oldest_available,
        } => {
            writer.put_u8(4);
            writer.put_u128(target.raw());
            writer.put_u64(requested.raw());
            writer.put_u64(oldest_available.raw());
        }
        SyncError::MissingEntity { entity } => {
            writer.put_u8(5);
            writer.put_u128(entity.raw());
        }
        SyncError::DuplicateEntity { entity } => {
            writer.put_u8(6);
            writer.put_u128(entity.raw());
        }
        SyncError::MissingComponent { entity, component } => {
            writer.put_u8(7);
            writer.put_u128(entity.raw());
            writer.put_string(component.as_str())?;
        }
        SyncError::MissingProperty { path } => {
            writer.put_u8(8);
            encode_path(writer, path)?;
        }
        SyncError::InvalidValueShape { path, detail } => {
            writer.put_u8(9);
            encode_path(writer, path)?;
            writer.put_string(detail)?;
        }
        SyncError::TargetMismatch { expected, actual } => {
            writer.put_u8(10);
            writer.put_u128(expected.raw());
            writer.put_u128(actual.raw());
        }
        SyncError::RevisionGap { expected, actual } => {
            writer.put_u8(11);
            writer.put_u64(expected.raw());
            writer.put_u64(actual.raw());
        }
    }
    Ok(())
}

fn decode_error(reader: &mut Reader<'_>) -> Result<SyncError, CodecError> {
    match reader.read_u8()? {
        0 => Ok(SyncError::InvalidRequest {
            code: decode_error_code(reader)?,
            detail: reader.read_string()?,
        }),
        1 => Ok(SyncError::UnsupportedCapability {
            capability: decode_capability(reader.read_u8()?)?,
        }),
        2 => Ok(SyncError::UnknownTarget {
            target: TargetId::from_raw(reader.read_u128()?),
        }),
        3 => Ok(SyncError::RevisionConflict {
            target: TargetId::from_raw(reader.read_u128()?),
            expected: Revision::from_raw(reader.read_u64()?),
            actual: Revision::from_raw(reader.read_u64()?),
        }),
        4 => Ok(SyncError::RevisionUnavailable {
            target: TargetId::from_raw(reader.read_u128()?),
            requested: Revision::from_raw(reader.read_u64()?),
            oldest_available: Revision::from_raw(reader.read_u64()?),
        }),
        5 => Ok(SyncError::MissingEntity {
            entity: EntityId::from_raw(reader.read_u128()?),
        }),
        6 => Ok(SyncError::DuplicateEntity {
            entity: EntityId::from_raw(reader.read_u128()?),
        }),
        7 => Ok(SyncError::MissingComponent {
            entity: EntityId::from_raw(reader.read_u128()?),
            component: ComponentId::new(reader.read_string()?).map_err(sync_to_codec)?,
        }),
        8 => Ok(SyncError::MissingProperty {
            path: decode_path(reader)?,
        }),
        9 => Ok(SyncError::InvalidValueShape {
            path: decode_path(reader)?,
            detail: reader.read_string()?,
        }),
        10 => Ok(SyncError::TargetMismatch {
            expected: TargetId::from_raw(reader.read_u128()?),
            actual: TargetId::from_raw(reader.read_u128()?),
        }),
        11 => Ok(SyncError::RevisionGap {
            expected: Revision::from_raw(reader.read_u64()?),
            actual: Revision::from_raw(reader.read_u64()?),
        }),
        tag => Err(CodecError::InvalidTag { kind: "error", tag }),
    }
}

fn encode_error_code(writer: &mut Writer, code: ErrorCode) {
    writer.put_u8(error_code_tag(code));
}

fn decode_error_code(reader: &mut Reader<'_>) -> Result<ErrorCode, CodecError> {
    match reader.read_u8()? {
        0 => Ok(ErrorCode::InvalidRequest),
        1 => Ok(ErrorCode::InvalidIdentifier),
        2 => Ok(ErrorCode::InvalidPropertyPath),
        3 => Ok(ErrorCode::DuplicateKey),
        4 => Ok(ErrorCode::InvalidCapabilityBits),
        5 => Ok(ErrorCode::UnsupportedCapability),
        6 => Ok(ErrorCode::UnknownTarget),
        7 => Ok(ErrorCode::RevisionConflict),
        8 => Ok(ErrorCode::RevisionUnavailable),
        9 => Ok(ErrorCode::MissingEntity),
        10 => Ok(ErrorCode::DuplicateEntity),
        11 => Ok(ErrorCode::MissingComponent),
        12 => Ok(ErrorCode::MissingProperty),
        13 => Ok(ErrorCode::InvalidValueShape),
        14 => Ok(ErrorCode::InvalidRevisionRange),
        15 => Ok(ErrorCode::TargetMismatch),
        16 => Ok(ErrorCode::RevisionGap),
        tag => Err(CodecError::InvalidTag {
            kind: "error code",
            tag,
        }),
    }
}

fn error_code_tag(code: ErrorCode) -> u8 {
    match code {
        ErrorCode::InvalidRequest => 0,
        ErrorCode::InvalidIdentifier => 1,
        ErrorCode::InvalidPropertyPath => 2,
        ErrorCode::DuplicateKey => 3,
        ErrorCode::InvalidCapabilityBits => 4,
        ErrorCode::UnsupportedCapability => 5,
        ErrorCode::UnknownTarget => 6,
        ErrorCode::RevisionConflict => 7,
        ErrorCode::RevisionUnavailable => 8,
        ErrorCode::MissingEntity => 9,
        ErrorCode::DuplicateEntity => 10,
        ErrorCode::MissingComponent => 11,
        ErrorCode::MissingProperty => 12,
        ErrorCode::InvalidValueShape => 13,
        ErrorCode::InvalidRevisionRange => 14,
        ErrorCode::TargetMismatch => 15,
        ErrorCode::RevisionGap => 16,
    }
}

fn capability_tag(capability: Capability) -> u8 {
    match capability {
        Capability::SnapshotRead => 0,
        Capability::IncrementalRead => 1,
        Capability::ComponentWrite => 2,
        Capability::PropertyWrite => 3,
        Capability::RevisionChecks => 4,
        Capability::DeterministicCodec => 5,
    }
}

fn decode_capability(tag: u8) -> Result<Capability, CodecError> {
    match tag {
        0 => Ok(Capability::SnapshotRead),
        1 => Ok(Capability::IncrementalRead),
        2 => Ok(Capability::ComponentWrite),
        3 => Ok(Capability::PropertyWrite),
        4 => Ok(Capability::RevisionChecks),
        5 => Ok(Capability::DeterministicCodec),
        tag => Err(CodecError::InvalidTag {
            kind: "capability",
            tag,
        }),
    }
}

fn sync_to_codec(error: SyncError) -> CodecError {
    CodecError::InvalidModel(error.to_string())
}

struct Writer {
    bytes: Vec<u8>,
    limits: CodecLimits,
}

impl Writer {
    fn new(limits: CodecLimits) -> Self {
        Self {
            bytes: Vec::new(),
            limits,
        }
    }

    fn finish(self) -> Result<Vec<u8>, CodecError> {
        if self.bytes.len() > self.limits.max_frame_bytes {
            return Err(CodecError::LimitExceeded {
                kind: "frame",
                limit: self.limits.max_frame_bytes,
            });
        }
        Ok(self.bytes)
    }

    fn put_bytes_raw(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn put_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn put_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u128(&mut self, value: u128) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_count(&mut self, count: usize, kind: &'static str) -> Result<(), CodecError> {
        if count > self.limits.max_collection_items {
            return Err(CodecError::LimitExceeded {
                kind,
                limit: self.limits.max_collection_items,
            });
        }
        let count = u32::try_from(count).map_err(|_| CodecError::LengthOverflow { kind })?;
        self.put_u32(count);
        Ok(())
    }

    fn put_string(&mut self, value: &str) -> Result<(), CodecError> {
        if value.len() > self.limits.max_string_bytes {
            return Err(CodecError::LimitExceeded {
                kind: "string",
                limit: self.limits.max_string_bytes,
            });
        }
        self.put_length(value.len(), "string")?;
        self.put_bytes_raw(value.as_bytes());
        Ok(())
    }

    fn put_bytes(&mut self, value: &[u8]) -> Result<(), CodecError> {
        if value.len() > self.limits.max_bytes {
            return Err(CodecError::LimitExceeded {
                kind: "byte array",
                limit: self.limits.max_bytes,
            });
        }
        self.put_length(value.len(), "byte array")?;
        self.put_bytes_raw(value);
        Ok(())
    }

    fn put_length(&mut self, length: usize, kind: &'static str) -> Result<(), CodecError> {
        let length = u32::try_from(length).map_err(|_| CodecError::LengthOverflow { kind })?;
        self.put_u32(length);
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
    limits: CodecLimits,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], limits: CodecLimits) -> Result<Self, CodecError> {
        if bytes.len() > limits.max_frame_bytes {
            return Err(CodecError::LimitExceeded {
                kind: "frame",
                limit: limits.max_frame_bytes,
            });
        }
        Ok(Self {
            bytes,
            offset: 0,
            limits,
        })
    }

    fn read_magic(&mut self) -> Result<(), CodecError> {
        if self.read_exact(4)? != MAGIC {
            return Err(CodecError::InvalidMagic);
        }
        Ok(())
    }

    fn read_exact(&mut self, length: usize) -> Result<&'a [u8], CodecError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(CodecError::LengthOverflow { kind: "field" })?;
        if end > self.bytes.len() {
            return Err(CodecError::Truncated);
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, CodecError> {
        Ok(u16::from_le_bytes(
            self.read_exact(2)?.try_into().expect("length checked"),
        ))
    }

    fn read_u32(&mut self) -> Result<u32, CodecError> {
        Ok(u32::from_le_bytes(
            self.read_exact(4)?.try_into().expect("length checked"),
        ))
    }

    fn read_u64(&mut self) -> Result<u64, CodecError> {
        Ok(u64::from_le_bytes(
            self.read_exact(8)?.try_into().expect("length checked"),
        ))
    }

    fn read_u128(&mut self) -> Result<u128, CodecError> {
        Ok(u128::from_le_bytes(
            self.read_exact(16)?.try_into().expect("length checked"),
        ))
    }

    fn read_i64(&mut self) -> Result<i64, CodecError> {
        Ok(i64::from_le_bytes(
            self.read_exact(8)?.try_into().expect("length checked"),
        ))
    }

    fn read_count(&mut self, kind: &'static str) -> Result<usize, CodecError> {
        let count = self.read_u32()? as usize;
        if count > self.limits.max_collection_items {
            return Err(CodecError::LimitExceeded {
                kind,
                limit: self.limits.max_collection_items,
            });
        }
        Ok(count)
    }

    fn read_string(&mut self) -> Result<String, CodecError> {
        let length = self.read_u32()? as usize;
        if length > self.limits.max_string_bytes {
            return Err(CodecError::LimitExceeded {
                kind: "string",
                limit: self.limits.max_string_bytes,
            });
        }
        String::from_utf8(self.read_exact(length)?.to_vec()).map_err(|_| CodecError::InvalidUtf8)
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, CodecError> {
        let length = self.read_u32()? as usize;
        if length > self.limits.max_bytes {
            return Err(CodecError::LimitExceeded {
                kind: "byte array",
                limit: self.limits.max_bytes,
            });
        }
        Ok(self.read_exact(length)?.to_vec())
    }

    fn finish(&self) -> Result<(), CodecError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(CodecError::TrailingBytes {
                count: self.bytes.len() - self.offset,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Capability, Change, ComponentId, EndpointCapabilities, EntityId, EntityRecord,
        PropertyPath, PropertySegment,
    };

    fn sample_request() -> RequestEnvelope {
        let path =
            PropertyPath::new(vec![PropertySegment::Field(String::from("x"))]).expect("path");
        RequestEnvelope {
            request_id: RequestId::from_raw(42),
            request: Request::ApplyChanges {
                target: TargetId::from_raw(7),
                base_revision: Revision::from_raw(3),
                changes: vec![Change::SetProperty {
                    entity: EntityId::from_raw(9),
                    component: ComponentId::from_static("Transform"),
                    path,
                    value: Value::Object(BTreeMap::from([
                        (String::from("a"), Value::Float(f64::NAN.to_bits())),
                        (String::from("b"), Value::Bytes(vec![1, 2, 3])),
                    ])),
                }],
            },
        }
    }

    #[test]
    fn request_round_trip_is_exact_and_deterministic() {
        let request = sample_request();
        let first = encode_request(&request).expect("encode");
        let second = encode_request(&request).expect("encode");
        assert_eq!(first, second);
        assert_eq!(decode_request(&first).expect("decode"), request);
    }

    #[test]
    fn response_round_trip_covers_snapshot_and_descriptor() {
        let mut snapshot = WorldSnapshot::empty(TargetId::from_raw(1));
        snapshot
            .entities
            .insert(EntityId::from_raw(2), EntityRecord::new());
        let response = ResponseEnvelope {
            request_id: RequestId::from_raw(3),
            response: Response::Success(ResponsePayload::Snapshot(snapshot)),
        };
        let bytes = encode_response(&response).expect("encode");
        assert_eq!(decode_response(&bytes).expect("decode"), response);

        let descriptor = EndpointDescriptor {
            protocol: ProtocolVersion::CURRENT,
            name: String::from("runtime"),
            capabilities: EndpointCapabilities::from_capabilities([
                Capability::SnapshotRead,
                Capability::DeterministicCodec,
            ]),
            targets: vec![TargetId::from_raw(1)],
        };
        let response = ResponseEnvelope {
            request_id: RequestId::from_raw(4),
            response: Response::Success(ResponsePayload::Descriptor(descriptor)),
        };
        let bytes = encode_response(&response).expect("encode");
        assert_eq!(decode_response(&bytes).expect("decode"), response);
    }

    #[test]
    fn error_round_trip_and_trailing_bytes_are_explicit() {
        let response = ResponseEnvelope {
            request_id: RequestId::from_raw(8),
            response: Response::Error(SyncError::RevisionConflict {
                target: TargetId::from_raw(1),
                expected: Revision::from_raw(4),
                actual: Revision::from_raw(3),
            }),
        };
        let mut bytes = encode_response(&response).expect("encode");
        assert_eq!(decode_response(&bytes).expect("decode"), response);
        bytes.push(0);
        assert!(matches!(
            decode_response(&bytes),
            Err(CodecError::TrailingBytes { count: 1 })
        ));
    }

    #[test]
    fn decoder_limits_untrusted_values_before_allocation() {
        let request = RequestEnvelope {
            request_id: RequestId::from_raw(1),
            request: Request::Ping,
        };
        let bytes = encode_request(&request).expect("encode");
        let limits = CodecLimits {
            max_frame_bytes: bytes.len() - 1,
            ..CodecLimits::default()
        };
        assert!(matches!(
            decode_request_with_limits(&bytes, limits),
            Err(CodecError::LimitExceeded { kind: "frame", .. })
        ));
    }
}
