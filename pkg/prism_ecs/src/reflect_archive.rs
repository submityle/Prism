//! Reflected component and prefab archives (design §16.3 / §24.3).
//!
//! V3 writes a closed prefab graph: requested nonempty entities followed by
//! their ancestors in breadth-first discovery order. Empty ancestors and pure
//! instances are retained. Repeated roots are deduplicated; dead entities,
//! dangling parents and cycles are errors. Parent order is inheritance priority.
//!
//! All integers below are little endian; strings and blobs are length prefixed.
//! A type reference is a u16 UTF-8 name length, name, then u64 StableTypeId.
//! A blob/string32 is a u32 byte length followed by those bytes.
//!
//! ```text
//! PECA | version:u8 | entities:u32
//! entity: source:u64 | components:u32
//!   component: type-reference | blob(prism_reflect binary)
//!   parents:u32 | source-parent:u64 ...
//!   policies:u32
//!     policy: component-type-reference | blob(schema-layout) | tag:u8
//!       tag 0: Removed
//!       tag 1: fields:u32
//!         field: string32(canonical-path) | string32(original-path)
//!                | field-type-reference | blob(prism_reflect binary)
//! ```
//!
//! V2 has the identical component records but no parent/policy sections, and
//! remains readable. V1's nameless format is refused. Components still use the
//! registered schema migration chain. Field policies require an exact canonical
//! schema layout; a component migration cannot infer how field ownership moves.
//! Dormant policies are checked against that schema even when no base exists.
//!
//! Loading defaults to reusing ids that were live before loading, otherwise
//! spawning distinct entities. Options can force fresh ids; the returned map
//! identifies every record, including automatically included ancestors. IsA
//! references always use this map. Optional remapping callbacks handle explicit
//! entity references in application component/field values: integer values are
//! never guessed to be entity ids.
//!
//! V3 replaces the reflected component set, outgoing IsA edges and policies of
//! each landed entity. Nonreflected components are retained. V2 retains its
//! original merge semantics and leaves prefab state alone. Parsing, migrations,
//! graph/schema checks and FromReflect preflight precede destination writes.
//! Remapping errors discard newly spawned entities before applying any records.
//! Arbitrary user hooks/panics are not a transactional storage boundary.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::entity::Entity;
use crate::prefab::reflect::{
    resolve_prefab_layers, PrefabComponentOverride, PrefabFieldOverride, PrefabOverrides,
    PrefabReflectError,
};
use crate::prefab::{IsA, PrefabError};
use crate::world::World;
use prism_reflect::{
    Access, DeserializeError, DeserializeLimits, DynamicArray, DynamicEnum, DynamicList,
    DynamicMap, DynamicSet, DynamicStruct, DynamicTupleStruct, DynamicVariant, MigrateError,
    ParsedPath, Patch, Reflect, ReflectRef, SerializeError, StableTypeId, TypeInfo,
    TypeRegistration, TypeRegistry, VariantKind, VariantType,
};

const MAGIC: [u8; 4] = *b"PECA";
const VERSION: u8 = 3;

/// Failure to encode, validate or materialize an archive.
#[derive(Debug)]
pub enum ReflectArchiveError {
    /// Missing archive magic.
    BadMagic,
    /// Unsupported envelope version.
    UnsupportedVersion(u8),
    /// Incomplete record.
    Truncated,
    /// Invalid generational entity handle.
    InvalidEntity(u64),
    /// Concrete name and stable id disagree.
    NameIdMismatch(String),
    /// Invalid UTF-8 in a name or path.
    MalformedName,
    /// Component is not registered in the destination bridge.
    UnknownType(StableTypeId),
    /// Registered schema migration failed.
    Migrate(MigrateError),
    /// Reflection encoding failed.
    Encode(SerializeError),
    /// Reflection decoding failed.
    Decode(DeserializeError),
    /// Output exceeds an envelope length field.
    TooLarge,
    /// Configured input or validation budget exceeded.
    LimitExceeded(&'static str),
    /// FromReflect rejected a decoded component.
    Unrepresentable(StableTypeId),
    /// Repeated source entity.
    DuplicateEntity(Entity),
    /// Repeated component (including migration collisions).
    DuplicateComponent(StableTypeId),
    /// A parent is not represented by an archive record.
    MissingPrefabBase(Entity),
    /// Policy schema differs from the currently registered component.
    PrefabSchemaMismatch(StableTypeId),
    /// Malformed policy envelope.
    InvalidPrefabMetadata(&'static str),
    /// Invalid ancestry or reflected field.
    Prefab(PrefabReflectError),
    /// Bytes remain after the last record.
    TrailingData,
}

impl core::fmt::Display for ReflectArchiveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => f.write_str("not a Prism reflect archive (bad magic)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported archive version {v}"),
            Self::Truncated => f.write_str("archive ended mid-record"),
            Self::InvalidEntity(bits) => write!(f, "invalid archive entity {bits:#x}"),
            Self::NameIdMismatch(name) => write!(f, "archive stable id disagrees with {name}"),
            Self::MalformedName => f.write_str("archive name or path is not UTF-8"),
            Self::UnknownType(id) => write!(f, "unregistered archive component {id:?}"),
            Self::Migrate(e) => write!(f, "schema migration failed: {e}"),
            Self::Encode(e) => write!(f, "reflection encode failed: {e}"),
            Self::Decode(e) => write!(f, "reflection decode failed: {e}"),
            Self::TooLarge => f.write_str("archive exceeds its length fields"),
            Self::LimitExceeded(limit) => write!(f, "archive exceeds {limit} budget"),
            Self::Unrepresentable(id) => write!(f, "component {id:?} cannot be reconstructed"),
            Self::DuplicateEntity(e) => write!(f, "duplicate archive entity {e:?}"),
            Self::DuplicateComponent(id) => write!(f, "duplicate archive component {id:?}"),
            Self::MissingPrefabBase(e) => write!(f, "archive omits prefab base {e:?}"),
            Self::PrefabSchemaMismatch(id) => write!(f, "prefab policy schema changed for {id:?}"),
            Self::InvalidPrefabMetadata(reason) => write!(f, "invalid prefab metadata: {reason}"),
            Self::Prefab(e) => e.fmt(f),
            Self::TrailingData => f.write_str("archive has trailing bytes"),
        }
    }
}
impl core::error::Error for ReflectArchiveError {}
impl From<SerializeError> for ReflectArchiveError {
    fn from(e: SerializeError) -> Self {
        Self::Encode(e)
    }
}
impl From<DeserializeError> for ReflectArchiveError {
    fn from(e: DeserializeError) -> Self {
        Self::Decode(e)
    }
}
impl From<MigrateError> for ReflectArchiveError {
    fn from(e: MigrateError) -> Self {
        Self::Migrate(e)
    }
}
impl From<PrefabReflectError> for ReflectArchiveError {
    fn from(e: PrefabReflectError) -> Self {
        Self::Prefab(e)
    }
}
impl From<PrefabError> for ReflectArchiveError {
    fn from(e: PrefabError) -> Self {
        Self::Prefab(e.into())
    }
}

/// Aggregate envelope limits, in addition to reflection's per-value limits.
#[derive(Clone, Copy, Debug)]
pub struct ReflectArchiveLimits {
    /// Maximum input bytes.
    pub max_bytes: usize,
    /// Maximum entity records.
    pub max_entities: usize,
    /// Maximum total components, edges, policies and field records.
    pub max_records: usize,
    /// Maximum path segments (also bounds recursive field replacement).
    pub max_path_depth: usize,
    /// Maximum graph visits and field/schema validation steps.
    pub max_validation_steps: usize,
    /// Reflection decoder depth and collection limits for each value.
    pub values: DeserializeLimits,
}
impl Default for ReflectArchiveLimits {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            max_entities: 1_000_000,
            max_records: 4_000_000,
            max_path_depth: 128,
            max_validation_steps: 8_000_000,
            values: DeserializeLimits::DEFAULT,
        }
    }
}

/// Explicit application reference remapping. Called on each staged component
/// (path None) and each selected override subtree (path Some). A callback must
/// preserve the value's reflected schema and return an error for missing
/// references according to the application's policy.
pub type ReflectArchiveRemapFn = fn(
    StableTypeId,
    Option<&ParsedPath>,
    &mut dyn Reflect,
    &BTreeMap<Entity, Entity>,
) -> Result<(), ReflectArchiveError>;

/// Load policy; default behavior matches the historical id reuse rule.
#[derive(Default)]
pub struct ReflectArchiveLoadOptions<'a> {
    /// Allocate fresh entities even when source ids already exist.
    pub spawn_fresh: bool,
    /// Additional nested reflected types. World component schemas take priority.
    pub types: Option<&'a TypeRegistry>,
    /// Optional explicit remapping of references inside values.
    pub remap: Option<ReflectArchiveRemapFn>,
    /// Envelope and reflected-value budgets.
    pub limits: ReflectArchiveLimits,
}

fn push_u16(out: &mut Vec<u8>, n: u16) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn push_u32(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn push_u64(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn push_len(out: &mut Vec<u8>, n: usize) -> Result<(), ReflectArchiveError> {
    push_u32(
        out,
        u32::try_from(n).map_err(|_| ReflectArchiveError::TooLarge)?,
    );
    Ok(())
}
fn push_blob(out: &mut Vec<u8>, blob: &[u8]) -> Result<(), ReflectArchiveError> {
    push_len(out, blob.len())?;
    out.extend_from_slice(blob);
    Ok(())
}
fn push_type(out: &mut Vec<u8>, name: &str, id: StableTypeId) -> Result<(), ReflectArchiveError> {
    if StableTypeId::of_path(name) != id {
        return Err(ReflectArchiveError::NameIdMismatch(name.into()));
    }
    push_u16(
        out,
        u16::try_from(name.len()).map_err(|_| ReflectArchiveError::TooLarge)?,
    );
    out.extend_from_slice(name.as_bytes());
    push_u64(out, id.value());
    Ok(())
}
fn spend(
    remaining: &mut usize,
    count: usize,
    what: &'static str,
) -> Result<(), ReflectArchiveError> {
    *remaining = remaining
        .checked_sub(count)
        .ok_or(ReflectArchiveError::LimitExceeded(what))?;
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    records: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], ReflectArchiveError> {
        let end = self
            .at
            .checked_add(len)
            .filter(|&n| n <= self.bytes.len())
            .ok_or(ReflectArchiveError::Truncated)?;
        let value = &self.bytes[self.at..end];
        self.at = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, ReflectArchiveError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, ReflectArchiveError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }
    fn u32(&mut self) -> Result<u32, ReflectArchiveError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, ReflectArchiveError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn entity(&mut self) -> Result<Entity, ReflectArchiveError> {
        let bits = self.u64()?;
        Entity::from_bits(bits).ok_or(ReflectArchiveError::InvalidEntity(bits))
    }
    fn text(&mut self, n: usize) -> Result<&'a str, ReflectArchiveError> {
        core::str::from_utf8(self.take(n)?).map_err(|_| ReflectArchiveError::MalformedName)
    }
    fn blob(&mut self) -> Result<&'a [u8], ReflectArchiveError> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    fn path(&mut self, limits: &ReflectArchiveLimits) -> Result<&'a str, ReflectArchiveError> {
        let n = self.u32()? as usize;
        // A long bare token can allocate heavily before the segment count is known.
        if n > 64 * 1024 {
            return Err(ReflectArchiveError::LimitExceeded("path bytes"));
        }
        let text = self.text(n)?;
        let path = ParsedPath::parse(text).map_err(PrefabReflectError::InvalidPath)?;
        if path.segments().len() > limits.max_path_depth {
            return Err(ReflectArchiveError::LimitExceeded("path depth"));
        }
        if path.is_empty() {
            return Err(PrefabReflectError::EmptyPath.into());
        }
        Ok(text)
    }
    fn type_ref(&mut self) -> Result<(&'a str, StableTypeId), ReflectArchiveError> {
        let n = self.u16()? as usize;
        let name = self.text(n)?;
        let id = StableTypeId::from_raw(self.u64()?);
        if StableTypeId::of_path(name) != id {
            return Err(ReflectArchiveError::NameIdMismatch(name.into()));
        }
        Ok((name, id))
    }
    fn count(&mut self, minimum_bytes: usize) -> Result<usize, ReflectArchiveError> {
        let n = self.u32()? as usize;
        if n > (self.bytes.len() - self.at) / minimum_bytes {
            return Err(ReflectArchiveError::Truncated);
        }
        spend(&mut self.records, n, "record count")?;
        Ok(n)
    }
}

struct ArchivedEntity {
    source: Entity,
    components: BTreeMap<StableTypeId, Box<dyn Reflect>>,
    parents: Vec<Entity>,
    overrides: PrefabOverrides,
}

// Decoder-only registry: runtime entries are indexed by name, not the shared
// sentinel TypeId. Registering the authoritative world entries last prevents
// supplemental schemas from replacing concrete component definitions.
fn archive_types(world: &World, extra: Option<&TypeRegistry>) -> TypeRegistry {
    let mut types = TypeRegistry::new();
    macro_rules! primitives { ($($t:ty),*) => { $(types.register::<$t>();)* }; }
    primitives!(
        bool, char, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64,
        String
    );
    for registry in extra
        .into_iter()
        .chain(core::iter::once(world.reflect.types()))
    {
        for r in registry.iter() {
            types.register_runtime(TypeRegistration::runtime(r.type_name(), r.type_info()));
        }
    }
    types
}

fn info(types: &TypeRegistry, name: &str) -> Result<&'static TypeInfo, ReflectArchiveError> {
    types
        .get_with_name(name)
        .map(|r| r.type_info())
        .ok_or_else(|| DeserializeError::UnregisteredType(name.into()).into())
}

// Binary reflection omits field names. Normalize dynamic values to declaration
// order before encoding; merely decoding the resulting bytes cannot detect a
// permutation of two fields with the same type. This also checks nested types
// supplied by a remapping callback, including empty-container schemas.
fn canonical_value(
    value: &dyn Reflect,
    schema: &'static TypeInfo,
    types: &TypeRegistry,
    work: &mut usize,
    limits: &DeserializeLimits,
    depth: usize,
) -> Result<Box<dyn Reflect>, ReflectArchiveError> {
    spend(work, 1, "validation steps")?;
    if depth > limits.max_depth {
        return Err(DeserializeError::DepthLimitExceeded {
            limit: limits.max_depth,
        }
        .into());
    }
    if value.type_name() != schema.type_name() {
        return Err(PrefabReflectError::FieldTypeMismatch {
            expected: schema.type_name(),
            actual: value.type_name(),
        }
        .into());
    }
    let invalid =
        || ReflectArchiveError::Prefab(PrefabReflectError::InvalidFieldShape(schema.type_name()));
    let collection = |len: usize| -> Result<(), ReflectArchiveError> {
        if len > limits.max_collection_len {
            return Err(DeserializeError::CollectionTooLarge {
                limit: limits.max_collection_len,
                declared: len,
            }
            .into());
        }
        Ok(())
    };
    let mut child = |value: &dyn Reflect, name: &str| {
        canonical_value(value, info(types, name)?, types, work, limits, depth + 1)
    };
    let mut result: Box<dyn Reflect> = match (schema, value.reflect_ref()) {
        (TypeInfo::Struct(s), ReflectRef::Struct(v)) if s.fields().len() == v.field_count() => {
            let mut out = DynamicStruct::new();
            for field in s.fields() {
                out.insert_boxed(
                    field.name(),
                    child(
                        v.field(field.name()).ok_or_else(invalid)?,
                        field.type_name(),
                    )?,
                );
            }
            Box::new(out)
        }
        (TypeInfo::TupleStruct(s), ReflectRef::TupleStruct(v))
            if s.fields().len() == v.field_count() =>
        {
            let mut out = DynamicTupleStruct::new();
            for field in s.fields() {
                out.insert_boxed(child(
                    v.field(field.index()).ok_or_else(invalid)?,
                    field.type_name(),
                )?);
            }
            Box::new(out)
        }
        (TypeInfo::Enum(s), ReflectRef::Enum(v)) => {
            let variant = s.variant(v.variant_name()).ok_or_else(invalid)?;
            if variant.index() != v.variant_index() || variant.field_count() != v.field_count() {
                return Err(invalid());
            }
            let payload = match (variant.kind(), v.variant_type()) {
                (VariantKind::Unit, VariantType::Unit) => DynamicVariant::Unit,
                (VariantKind::Tuple(fields), VariantType::Tuple) => DynamicVariant::Tuple(
                    fields
                        .iter()
                        .map(|f| child(v.field_at(f.index()).ok_or_else(invalid)?, f.type_name()))
                        .collect::<Result<_, _>>()?,
                ),
                (VariantKind::Struct(fields), VariantType::Struct) => DynamicVariant::Struct(
                    fields
                        .iter()
                        .map(|f| {
                            Ok((
                                f.name(),
                                child(v.field(f.name()).ok_or_else(invalid)?, f.type_name())?,
                            ))
                        })
                        .collect::<Result<_, ReflectArchiveError>>()?,
                ),
                _ => return Err(invalid()),
            };
            Box::new(DynamicEnum::new(variant.index(), variant.name(), payload))
        }
        (TypeInfo::List(s), ReflectRef::List(v)) => {
            collection(v.len())?;
            let mut out = DynamicList::new();
            for item in v.iter_reflect() {
                out.push_boxed(child(item, s.item_type_name())?);
            }
            Box::new(out)
        }
        (TypeInfo::Array(s), ReflectRef::Array(v)) if s.capacity() == v.len() => {
            collection(v.len())?;
            let mut out = DynamicArray::new();
            for i in 0..v.len() {
                out.push_boxed(child(v.get(i).ok_or_else(invalid)?, s.item_type_name())?);
            }
            Box::new(out)
        }
        (TypeInfo::Map(s), ReflectRef::Map(v)) => {
            collection(v.len())?;
            let mut out = DynamicMap::new();
            for (key, item) in v.iter_reflect() {
                out.insert_boxed(
                    child(key, s.key_type_name())?,
                    child(item, s.value_type_name())?,
                );
            }
            if prism_reflect::Map::len(&out) != v.len() {
                return Err(invalid());
            }
            Box::new(out)
        }
        (TypeInfo::Set(s), ReflectRef::Set(v)) => {
            collection(v.len())?;
            let mut out = DynamicSet::new();
            for item in v.iter_reflect() {
                if !out.push_boxed(child(item, s.value_type_name())?) {
                    return Err(invalid());
                }
            }
            Box::new(out)
        }
        (TypeInfo::Value(_), ReflectRef::Value(_)) => return Ok(value.reflect_clone()),
        _ => return Err(invalid()),
    };
    let name = schema.type_name();
    match schema {
        TypeInfo::Struct(_) => result
            .downcast_mut::<DynamicStruct>()
            .unwrap()
            .set_represented_type_name(name),
        TypeInfo::TupleStruct(_) => result
            .downcast_mut::<DynamicTupleStruct>()
            .unwrap()
            .set_represented_type_name(name),
        TypeInfo::Enum(_) => result
            .downcast_mut::<DynamicEnum>()
            .unwrap()
            .set_represented_type_name(name),
        TypeInfo::List(_) => result
            .downcast_mut::<DynamicList>()
            .unwrap()
            .set_represented_type_name(name),
        TypeInfo::Array(_) => result
            .downcast_mut::<DynamicArray>()
            .unwrap()
            .set_represented_type_name(name),
        TypeInfo::Map(_) => result
            .downcast_mut::<DynamicMap>()
            .unwrap()
            .set_represented_type_name(name),
        TypeInfo::Set(_) => result
            .downcast_mut::<DynamicSet>()
            .unwrap()
            .set_represented_type_name(name),
        _ => return Err(invalid()),
    }
    Ok(result)
}

fn value_blob(
    value: &dyn Reflect,
    schema: &'static TypeInfo,
    types: &TypeRegistry,
    work: &mut usize,
    limits: &DeserializeLimits,
) -> Result<Vec<u8>, ReflectArchiveError> {
    let value = canonical_value(value, schema, types, work, limits, 1)?;
    let bytes = prism_reflect::to_binary(&*value)?;
    prism_reflect::from_binary_with_limits(&bytes, types, schema, limits)?;
    Ok(bytes)
}

// Canonical schema bytes, not Debug output or an unstable hash. Each reachable
// type is described once, sorted by name, so recursive descriptors terminate.
fn schema_layout(
    root: &'static TypeInfo,
    types: &TypeRegistry,
    work: &mut usize,
) -> Result<Vec<u8>, ReflectArchiveError> {
    let mut pending = VecDeque::from([root]);
    let mut layouts = BTreeMap::new();
    while let Some(current) = pending.pop_front() {
        spend(work, 1, "validation steps")?;
        if layouts.contains_key(current.type_name()) {
            continue;
        }
        let mut out = Vec::new();
        let mut children = Vec::new();
        match current {
            TypeInfo::Struct(s) => {
                out.push(0);
                push_len(&mut out, s.fields().len())?;
                for f in s.fields() {
                    spend(work, 1, "validation steps")?;
                    push_blob(&mut out, f.name().as_bytes())?;
                    push_blob(&mut out, f.type_name().as_bytes())?;
                    children.push(f.type_name());
                }
            }
            TypeInfo::TupleStruct(s) => {
                out.push(1);
                push_len(&mut out, s.fields().len())?;
                for f in s.fields() {
                    spend(work, 1, "validation steps")?;
                    push_u64(&mut out, f.index() as u64);
                    push_blob(&mut out, f.type_name().as_bytes())?;
                    children.push(f.type_name());
                }
            }
            TypeInfo::Enum(e) => {
                out.push(2);
                push_len(&mut out, e.variants().len())?;
                for v in e.variants() {
                    spend(work, 1, "validation steps")?;
                    push_blob(&mut out, v.name().as_bytes())?;
                    push_u64(&mut out, v.index() as u64);
                    match v.kind() {
                        VariantKind::Unit => out.push(0),
                        VariantKind::Tuple(fields) => {
                            out.push(1);
                            push_len(&mut out, fields.len())?;
                            for f in fields {
                                spend(work, 1, "validation steps")?;
                                push_u64(&mut out, f.index() as u64);
                                push_blob(&mut out, f.type_name().as_bytes())?;
                                children.push(f.type_name());
                            }
                        }
                        VariantKind::Struct(fields) => {
                            out.push(2);
                            push_len(&mut out, fields.len())?;
                            for f in fields {
                                spend(work, 1, "validation steps")?;
                                push_blob(&mut out, f.name().as_bytes())?;
                                push_blob(&mut out, f.type_name().as_bytes())?;
                                children.push(f.type_name());
                            }
                        }
                        _ => {
                            return Err(ReflectArchiveError::InvalidPrefabMetadata(
                                "unsupported variant schema",
                            ))
                        }
                    }
                }
            }
            TypeInfo::List(l) => {
                out.push(3);
                children.push(l.item_type_name());
            }
            TypeInfo::Array(a) => {
                out.push(4);
                push_u64(&mut out, a.capacity() as u64);
                children.push(a.item_type_name());
            }
            TypeInfo::Map(m) => {
                out.push(5);
                children.push(m.key_type_name());
                children.push(m.value_type_name());
            }
            TypeInfo::Set(s) => {
                out.push(6);
                children.push(s.value_type_name());
            }
            TypeInfo::Value(_) => out.push(7),
            _ => {
                return Err(ReflectArchiveError::InvalidPrefabMetadata(
                    "unsupported type schema",
                ))
            }
        }
        // Include child names for containers as well as their recursive layouts.
        push_len(&mut out, children.len())?;
        for name in &children {
            push_blob(&mut out, name.as_bytes())?;
        }
        layouts.insert(current.type_name(), out);
        for name in children {
            pending.push_back(info(types, name)?);
        }
    }
    let mut out = Vec::new();
    push_len(&mut out, layouts.len())?;
    for (name, layout) in layouts {
        push_blob(&mut out, name.as_bytes())?;
        push_blob(&mut out, &layout)?;
    }
    Ok(out)
}

// Check both canonical ownership paths and the original spelling, including
// numeric aliases of named enum fields. Candidate types account for dormant
// overrides on variants that are not currently active.
fn field_info(
    root: &'static TypeInfo,
    canonical: &ParsedPath,
    original: &ParsedPath,
    field_name: &str,
    types: &TypeRegistry,
    work: &mut usize,
) -> Result<&'static TypeInfo, ReflectArchiveError> {
    if canonical.is_empty() || canonical.segments().len() != original.segments().len() {
        return Err(ReflectArchiveError::InvalidPrefabMetadata(
            "inconsistent field paths",
        ));
    }
    let mut candidates = BTreeMap::from([(root.type_name(), root)]);
    for (step, spelling) in canonical.segments().iter().zip(original.segments()) {
        let mut names = BTreeSet::new();
        for current in candidates.values() {
            spend(work, 1, "validation steps")?;
            match (*current, step, spelling) {
                (TypeInfo::Struct(s), Access::Field(name), Access::Field(alias))
                    if name == alias =>
                {
                    if let Some(f) = s.field(name) {
                        names.insert(f.type_name());
                    }
                }
                (TypeInfo::TupleStruct(s), Access::TupleIndex(i), Access::TupleIndex(j))
                    if i == j =>
                {
                    if let Some(f) = s.field_at(*i) {
                        names.insert(f.type_name());
                    }
                }
                (TypeInfo::Enum(e), Access::Field(name), alias) => {
                    for v in e.variants() {
                        spend(work, 1, "validation steps")?;
                        if let VariantKind::Struct(fields) = v.kind() {
                            for (i, f) in fields.iter().enumerate() {
                                spend(work, 1, "validation steps")?;
                                if f.name() == name
                                    && (alias == step || *alias == Access::TupleIndex(i))
                                {
                                    names.insert(f.type_name());
                                }
                            }
                        }
                    }
                }
                (TypeInfo::Enum(e), Access::TupleIndex(i), Access::TupleIndex(j)) if i == j => {
                    for v in e.variants() {
                        spend(work, 1, "validation steps")?;
                        if let VariantKind::Tuple(fields) = v.kind()
                            && let Some(f) = fields.get(*i)
                        {
                            names.insert(f.type_name());
                        }
                    }
                }
                (TypeInfo::List(l), Access::ListIndex(i), Access::ListIndex(j)) if i == j => {
                    names.insert(l.item_type_name());
                }
                (TypeInfo::Array(a), Access::ListIndex(i), Access::ListIndex(j))
                    if i == j && *i < a.capacity() =>
                {
                    names.insert(a.item_type_name());
                }
                (TypeInfo::Map(m), Access::Key(a), Access::Key(b))
                    if a == b && m.key_type_name() == core::any::type_name::<String>() =>
                {
                    names.insert(m.value_type_name());
                }
                _ => {}
            }
        }
        if names.is_empty() {
            return Err(PrefabReflectError::MissingField(alloc::format!("{canonical:?}")).into());
        }
        candidates = names
            .into_iter()
            .map(|n| Ok((n, info(types, n)?)))
            .collect::<Result<_, ReflectArchiveError>>()?;
    }
    candidates
        .get(field_name)
        .copied()
        .ok_or(ReflectArchiveError::InvalidPrefabMetadata(
            "field type does not match path schema",
        ))
}

fn validate_graph(
    records: &[ArchivedEntity],
    index: &BTreeMap<Entity, usize>,
    work: &mut usize,
) -> Result<(), ReflectArchiveError> {
    let mut done = BTreeSet::new();
    let mut active = BTreeSet::new();
    for record in records {
        let mut stack = alloc::vec![(record.source, false)];
        while let Some((entity, exiting)) = stack.pop() {
            spend(work, 1, "validation steps")?;
            if exiting {
                active.remove(&entity);
                done.insert(entity);
                continue;
            }
            if active.contains(&entity) {
                return Err(PrefabError::Cycle(entity).into());
            }
            if done.contains(&entity) {
                continue;
            }
            let i = *index
                .get(&entity)
                .ok_or(ReflectArchiveError::MissingPrefabBase(entity))?;
            active.insert(entity);
            stack.push((entity, true));
            for &parent in records[i].parents.iter().rev() {
                stack.push((parent, false));
            }
        }
    }
    Ok(())
}

fn validate_policies(
    world: &World,
    records: &[ArchivedEntity],
    index: &BTreeMap<Entity, usize>,
    types: &TypeRegistry,
    work: &mut usize,
    values: &DeserializeLimits,
) -> Result<(), ReflectArchiveError> {
    if !records
        .iter()
        .any(|r| r.overrides.components().next().is_some())
    {
        return Ok(());
    }
    for record in records {
        let mut queue = VecDeque::from([record.source]);
        let mut visited = BTreeSet::new();
        let mut chain = Vec::new();
        let mut ids = BTreeSet::new();
        while let Some(entity) = queue.pop_front() {
            spend(work, 1, "validation steps")?;
            if !visited.insert(entity) {
                continue;
            }
            let r = &records[index[&entity]];
            ids.extend(r.overrides.components().map(|(id, _)| id));
            chain.push(r);
            queue.extend(r.parents.iter().copied());
        }
        // A dormant policy on one ancestor can activate on a descendant with
        // another parent, so validate all reachable policy ids for every entity.
        for id in ids {
            spend(work, chain.len(), "validation steps")?;
            let layers = chain.iter().map(|r| {
                (
                    r.overrides.component(id),
                    r.components.get(&id).map(|v| &**v),
                )
            });
            if let Some((_, value)) = resolve_prefab_layers(layers)? {
                let root = component_info(world, id)?;
                value_blob(&*value, root, types, work, values)?;
            }
        }
    }
    Ok(())
}

fn component_info(
    world: &World,
    id: StableTypeId,
) -> Result<&'static TypeInfo, ReflectArchiveError> {
    if world.reflect.ops_for(id).is_none() {
        return Err(ReflectArchiveError::UnknownType(id));
    }
    world
        .reflect
        .types()
        .iter()
        .find(|r| StableTypeId::of_path(r.type_name()) == id)
        .map(|r| r.type_info())
        .ok_or(ReflectArchiveError::UnknownType(id))
}

fn validate_field(
    root: &'static TypeInfo,
    field: &PrefabFieldOverride,
    types: &TypeRegistry,
    work: &mut usize,
    limits: &ReflectArchiveLimits,
) -> Result<(), ReflectArchiveError> {
    if field.archive_path().len() > 64 * 1024 || field.path_str().len() > 64 * 1024 {
        return Err(ReflectArchiveError::LimitExceeded("path bytes"));
    }
    if field.path().segments().len() > limits.max_path_depth {
        return Err(ReflectArchiveError::LimitExceeded("path depth"));
    }
    let original = ParsedPath::parse(field.path_str()).map_err(PrefabReflectError::InvalidPath)?;
    let Patch::Replace(value) = field.patch() else {
        return Err(ReflectArchiveError::InvalidPrefabMetadata(
            "field is not a replacement",
        ));
    };
    let target = field_info(
        root,
        field.path(),
        &original,
        value.type_name(),
        types,
        work,
    )?;
    value_blob(&**value, target, types, work, &limits.values)?;
    Ok(())
}

impl World {
    /// Save concrete reflected components plus the complete prefab ancestor
    /// closure. Completely empty standalone roots are skipped.
    pub fn save_reflect_archive(
        &self,
        entities: impl IntoIterator<Item = Entity>,
    ) -> Result<Vec<u8>, ReflectArchiveError> {
        self.save_reflect_archive_with_types(entities, &TypeRegistry::new())
    }

    /// Save with registrations for nested structs/enums/containers. Primitive
    /// leaves need no extra registration; component root schemas come from World.
    pub fn save_reflect_archive_with_types(
        &self,
        entities: impl IntoIterator<Item = Entity>,
        extra: &TypeRegistry,
    ) -> Result<Vec<u8>, ReflectArchiveError> {
        let types = archive_types(self, Some(extra));
        let limits = ReflectArchiveLimits::default();
        let mut work = limits.max_validation_steps;
        let mut queue = VecDeque::new();
        let mut seen = BTreeSet::new();
        for entity in entities {
            spend(&mut work, 1, "validation steps")?;
            if !self.contains(entity) {
                return Err(PrefabError::MissingEntity(entity).into());
            }
            if (!self.reflect_entity(entity).is_empty()
                || !self.prefabs_of(entity).is_empty()
                || self
                    .get::<PrefabOverrides>(entity)
                    .is_some_and(|p| p.components().next().is_some()))
                && seen.insert(entity)
            {
                queue.push_back(entity);
            }
        }
        let mut records = Vec::new();
        let mut record_budget = limits.max_records;
        while let Some(entity) = queue.pop_front() {
            spend(&mut work, 1, "validation steps")?;
            if records.len() >= limits.max_entities {
                return Err(ReflectArchiveError::LimitExceeded("entity count"));
            }
            if !self.contains(entity) {
                return Err(PrefabError::MissingEntity(entity).into());
            }
            let parents = self.prefabs_of(entity).to_vec();
            let components = self
                .reflect_entity(entity)
                .into_iter()
                .map(|(id, value)| (id, value.reflect_clone()))
                .collect::<BTreeMap<_, _>>();
            let overrides = self
                .get::<PrefabOverrides>(entity)
                .cloned()
                .unwrap_or_default();
            spend(&mut record_budget, 1, "record count")?;
            spend(
                &mut record_budget,
                parents.len() + components.len(),
                "record count",
            )?;
            for &parent in &parents {
                if seen.insert(parent) {
                    queue.push_back(parent);
                }
            }
            for (id, policy) in overrides.components() {
                spend(&mut record_budget, 1, "record count")?;
                let root = component_info(self, id)?;
                schema_layout(root, &types, &mut work)?;
                if let PrefabComponentOverride::Fields(fields) = policy {
                    spend(&mut record_budget, fields.len(), "record count")?;
                    if fields.is_empty() {
                        return Err(ReflectArchiveError::InvalidPrefabMetadata(
                            "empty field policy",
                        ));
                    }
                    for field in fields {
                        validate_field(root, field, &types, &mut work, &limits)?;
                    }
                }
            }
            records.push(ArchivedEntity {
                source: entity,
                components,
                parents,
                overrides,
            });
        }
        let index = records
            .iter()
            .enumerate()
            .map(|(i, r)| (r.source, i))
            .collect();
        validate_graph(&records, &index, &mut work)?;
        validate_policies(self, &records, &index, &types, &mut work, &limits.values)?;
        let mut out = MAGIC.to_vec();
        out.push(VERSION);
        push_len(&mut out, records.len())?;
        for record in &records {
            push_u64(&mut out, record.source.to_bits());
            push_len(&mut out, record.components.len())?;
            for (&id, value) in &record.components {
                push_type(&mut out, value.type_name(), id)?;
                push_blob(
                    &mut out,
                    &value_blob(
                        &**value,
                        component_info(self, id)?,
                        &types,
                        &mut work,
                        &limits.values,
                    )?,
                )?;
            }
            push_len(&mut out, record.parents.len())?;
            for parent in &record.parents {
                push_u64(&mut out, parent.to_bits());
            }
            push_len(&mut out, record.overrides.components().count())?;
            for (id, policy) in record.overrides.components() {
                let root = component_info(self, id)?;
                push_type(&mut out, root.type_name(), id)?;
                push_blob(&mut out, &schema_layout(root, &types, &mut work)?)?;
                match policy {
                    PrefabComponentOverride::Removed => out.push(0),
                    PrefabComponentOverride::Fields(fields) => {
                        out.push(1);
                        push_len(&mut out, fields.len())?;
                        for field in fields {
                            push_blob(&mut out, field.archive_path().as_bytes())?;
                            push_blob(&mut out, field.path_str().as_bytes())?;
                            let Patch::Replace(value) = field.patch() else {
                                return Err(ReflectArchiveError::InvalidPrefabMetadata(
                                    "field is not a replacement",
                                ));
                            };
                            push_type(
                                &mut out,
                                value.type_name(),
                                StableTypeId::of_path(value.type_name()),
                            )?;
                            let original = ParsedPath::parse(field.path_str())
                                .map_err(PrefabReflectError::InvalidPath)?;
                            let target = field_info(
                                root,
                                field.path(),
                                &original,
                                value.type_name(),
                                &types,
                                &mut work,
                            )?;
                            push_blob(
                                &mut out,
                                &value_blob(&**value, target, &types, &mut work, &limits.values)?,
                            )?;
                        }
                    }
                }
            }
            if out.len() > limits.max_bytes {
                return Err(ReflectArchiveError::LimitExceeded("archive bytes"));
            }
        }
        Ok(out)
    }

    /// Load V2 or V3; returns landed entities in archive order, including bases.
    pub fn load_reflect_archive(
        &mut self,
        bytes: &[u8],
    ) -> Result<Vec<Entity>, ReflectArchiveError> {
        Ok(self
            .load_reflect_archive_with_options(bytes, ReflectArchiveLoadOptions::default())?
            .into_iter()
            .map(|(_, target)| target)
            .collect())
    }

    /// Load with explicit id allocation, extra nested schemas and reference
    /// remapping. Returns source/target pairs in archive order.
    pub fn load_reflect_archive_with_options(
        &mut self,
        bytes: &[u8],
        options: ReflectArchiveLoadOptions<'_>,
    ) -> Result<Vec<(Entity, Entity)>, ReflectArchiveError> {
        if bytes.len() > options.limits.max_bytes {
            return Err(ReflectArchiveError::LimitExceeded("archive bytes"));
        }
        let types = archive_types(self, options.types);
        let mut reader = Reader {
            bytes,
            at: 0,
            records: options.limits.max_records,
        };
        if reader.take(4)? != MAGIC {
            return Err(ReflectArchiveError::BadMagic);
        }
        let version = reader.u8()?;
        if version != 2 && version != VERSION {
            return Err(ReflectArchiveError::UnsupportedVersion(version));
        }
        let count = reader.count(if version == 2 { 12 } else { 20 })?;
        if count > options.limits.max_entities {
            return Err(ReflectArchiveError::LimitExceeded("entity count"));
        }
        let mut work = options.limits.max_validation_steps;
        let mut records = Vec::new();
        let mut index = BTreeMap::new();
        for i in 0..count {
            let source = reader.entity()?;
            if index.insert(source, i).is_some() {
                return Err(ReflectArchiveError::DuplicateEntity(source));
            }
            let mut components = BTreeMap::new();
            let mut written_ids = BTreeSet::new();
            for _ in 0..reader.count(14)? {
                let (name, id) = reader.type_ref()?;
                if !written_ids.insert(id) {
                    return Err(ReflectArchiveError::DuplicateComponent(id));
                }
                let blob = reader.blob()?;
                let (current_id, value) =
                    self.decode_archived_component(name, id, blob, &types, &options.limits.values)?;
                if components.insert(current_id, value).is_some() {
                    return Err(ReflectArchiveError::DuplicateComponent(current_id));
                }
            }
            let mut parents = Vec::new();
            let mut overrides = PrefabOverrides::default();
            if version == VERSION {
                let mut parent_set = BTreeSet::new();
                for _ in 0..reader.count(8)? {
                    let parent = reader.entity()?;
                    if !parent_set.insert(parent) {
                        return Err(ReflectArchiveError::InvalidPrefabMetadata(
                            "duplicate parent",
                        ));
                    }
                    parents.push(parent);
                }
                let mut policy_ids = BTreeSet::new();
                for _ in 0..reader.count(15)? {
                    let (name, id) = reader.type_ref()?;
                    if !policy_ids.insert(id) {
                        return Err(ReflectArchiveError::InvalidPrefabMetadata(
                            "duplicate policy",
                        ));
                    }
                    // Historical whole-component migrations are supported, but
                    // ownership paths require an explicit compatible schema.
                    if let Some((logical, wrote)) = self.reflect.schemas().locate(name)
                        && self
                            .reflect
                            .schemas()
                            .current_schema(logical)
                            .is_none_or(|s| s.type_name() != name || s.version() != wrote)
                    {
                        return Err(ReflectArchiveError::PrefabSchemaMismatch(id));
                    }
                    let root = component_info(self, id)?;
                    if root.type_name() != name
                        || reader.blob()? != schema_layout(root, &types, &mut work)?
                    {
                        return Err(ReflectArchiveError::PrefabSchemaMismatch(id));
                    }
                    let policy = match reader.u8()? {
                        0 => PrefabComponentOverride::Removed,
                        1 => {
                            let n = reader.count(22)?;
                            if n == 0 {
                                return Err(ReflectArchiveError::InvalidPrefabMetadata(
                                    "empty field policy",
                                ));
                            }
                            let mut fields = Vec::new();
                            for _ in 0..n {
                                let canonical = reader.path(&options.limits)?;
                                let spelling = reader.path(&options.limits)?;
                                let path = ParsedPath::parse(canonical)
                                    .map_err(PrefabReflectError::InvalidPath)?;
                                let original = ParsedPath::parse(spelling)
                                    .map_err(PrefabReflectError::InvalidPath)?;
                                let (field_name, _) = reader.type_ref()?;
                                let target = field_info(
                                    root, &path, &original, field_name, &types, &mut work,
                                )?;
                                let value = prism_reflect::from_binary_with_limits(
                                    reader.blob()?,
                                    &types,
                                    target,
                                    &options.limits.values,
                                )?;
                                // Bound pairwise overlap detection by the work budget.
                                spend(&mut work, fields.len(), "validation steps")?;
                                fields.push(PrefabFieldOverride::from_archive(
                                    canonical, spelling, value,
                                )?);
                            }
                            PrefabComponentOverride::Fields(fields)
                        }
                        _ => {
                            return Err(ReflectArchiveError::InvalidPrefabMetadata(
                                "unknown policy tag",
                            ))
                        }
                    };
                    overrides.insert_archived(id, policy)?;
                }
            }
            records.push(ArchivedEntity {
                source,
                components,
                parents,
                overrides,
            });
        }
        if reader.at != bytes.len() {
            return Err(ReflectArchiveError::TrailingData);
        }
        validate_graph(&records, &index, &mut work)?;
        validate_policies(
            self,
            &records,
            &index,
            &types,
            &mut work,
            &options.limits.values,
        )?;
        self.preflight_archive(&records)?;

        // Compute reuse for *all* sources before allocating. Otherwise the first
        // spawn can impersonate a later archived id and merge two records.
        let reuse: Vec<bool> = records
            .iter()
            .map(|r| !options.spawn_fresh && self.contains(r.source))
            .collect();
        let mut spawned = Vec::new();
        let landed: Vec<_> = records
            .iter()
            .zip(reuse)
            .map(|(r, reuse)| {
                let target = if reuse {
                    r.source
                } else {
                    let entity = self.spawn(());
                    spawned.push(entity);
                    entity
                };
                (r.source, target)
            })
            .collect();
        let mapping = landed.iter().copied().collect::<BTreeMap<_, _>>();

        if let Some(remap) = options.remap {
            let result = (|| -> Result<(), ReflectArchiveError> {
                for record in &mut records {
                    for (&id, value) in &mut record.components {
                        remap(id, None, &mut **value, &mapping)?;
                        // Callback output must still match its registered schema.
                        let root = component_info(self, id)?;
                        *value = canonical_value(
                            &**value,
                            root,
                            &types,
                            &mut work,
                            &options.limits.values,
                            1,
                        )?;
                        value_blob(&**value, root, &types, &mut work, &options.limits.values)?;
                    }
                    let mut updated = PrefabOverrides::default();
                    for (id, policy) in record.overrides.components() {
                        let policy = match policy {
                            PrefabComponentOverride::Removed => PrefabComponentOverride::Removed,
                            PrefabComponentOverride::Fields(fields) => {
                                let mut remapped = Vec::new();
                                for field in fields {
                                    let Patch::Replace(value) = field.patch() else {
                                        return Err(ReflectArchiveError::InvalidPrefabMetadata(
                                            "field is not a replacement",
                                        ));
                                    };
                                    let mut value = value.reflect_clone();
                                    remap(id, Some(field.path()), &mut *value, &mapping)?;
                                    let field = PrefabFieldOverride::from_archive(
                                        &field.archive_path(),
                                        field.path_str(),
                                        value,
                                    )?;
                                    validate_field(
                                        component_info(self, id)?,
                                        &field,
                                        &types,
                                        &mut work,
                                        &options.limits,
                                    )?;
                                    remapped.push(field);
                                }
                                PrefabComponentOverride::Fields(remapped)
                            }
                        };
                        updated.insert_archived(id, policy)?;
                    }
                    record.overrides = updated;
                }
                validate_policies(
                    self,
                    &records,
                    &index,
                    &types,
                    &mut work,
                    &options.limits.values,
                )?;
                self.preflight_archive(&records)
            })();
            if let Err(error) = result {
                for entity in spawned {
                    self.despawn(entity);
                }
                return Err(error);
            }
        }

        for (record, &(_, target)) in records.iter().zip(&landed) {
            if version == VERSION {
                let obsolete: Vec<_> = self
                    .reflect_entity(target)
                    .into_iter()
                    .map(|(id, _)| id)
                    .filter(|id| !record.components.contains_key(id))
                    .collect();
                for id in obsolete {
                    self.remove_reflect(target, id);
                }
            }
            for (&id, value) in &record.components {
                let ops = self
                    .reflect
                    .ops_for(id)
                    .ok_or(ReflectArchiveError::UnknownType(id))?;
                if !ops.apply_from_reflect(self, target, &**value) {
                    return Err(ReflectArchiveError::Unrepresentable(id));
                }
            }
        }
        if version == VERSION {
            // Remove every replaced outgoing edge before rebuilding, so old
            // relations cannot affect priority or introduce transient back edges.
            for &(_, target) in &landed {
                for parent in self.prefabs_of(target).to_vec() {
                    self.remove_relation::<IsA>(target, parent);
                }
            }
            for (record, &(_, target)) in records.iter().zip(&landed) {
                if record.overrides.components().next().is_some() {
                    self.insert(target, record.overrides.clone());
                } else {
                    self.remove::<PrefabOverrides>(target);
                }
                for parent in &record.parents {
                    self.add_prefab(target, mapping[parent]);
                }
            }
        }
        Ok(landed)
    }

    // Exercise FromReflect using the same bridge operations, before touching
    // destination entities. This also validates migrated concrete values.
    fn preflight_archive(&self, records: &[ArchivedEntity]) -> Result<(), ReflectArchiveError> {
        let mut scratch = World::new();
        let target = scratch.spawn(());
        for record in records {
            for (&id, value) in &record.components {
                let ops = self
                    .reflect
                    .ops_for(id)
                    .ok_or(ReflectArchiveError::UnknownType(id))?;
                if !ops.apply_from_reflect(&mut scratch, target, &**value) {
                    return Err(ReflectArchiveError::Unrepresentable(id));
                }
            }
        }
        Ok(())
    }

    fn decode_archived_component(
        &self,
        name: &str,
        type_id: StableTypeId,
        blob: &[u8],
        types: &TypeRegistry,
        limits: &DeserializeLimits,
    ) -> Result<(StableTypeId, Box<dyn Reflect>), ReflectArchiveError> {
        let resolved = self
            .reflect
            .schemas()
            .locate(name)
            .map(|(logical, version)| (logical.to_string(), version));
        let (id, decoded) = match resolved {
            Some((logical, wrote)) => {
                let schemas = self.reflect.schemas();
                let writer = schemas
                    .schema(&logical, wrote)
                    .ok_or(ReflectArchiveError::UnknownType(type_id))?;
                let mut decoded =
                    prism_reflect::from_binary_with_limits(blob, types, writer.info(), limits)?;
                for step in schemas.chain(&logical, wrote)? {
                    let value = decoded
                        .downcast_mut::<DynamicStruct>()
                        .ok_or(ReflectArchiveError::Unrepresentable(type_id))?;
                    step.apply(value)?;
                }
                let current = schemas
                    .current_schema(&logical)
                    .ok_or(ReflectArchiveError::UnknownType(type_id))?;
                // Migration functions edit fields but cannot change the
                // represented root name. Normalize it here, before policies or
                // remapping callbacks inspect the migrated component.
                if current.type_name() != name {
                    decoded
                        .downcast_mut::<DynamicStruct>()
                        .ok_or(ReflectArchiveError::Unrepresentable(type_id))?
                        .set_represented_type_name(current.type_name());
                }
                (StableTypeId::of_path(current.type_name()), decoded)
            }
            None => {
                let registration = self
                    .reflect
                    .types()
                    .get_with_name(name)
                    .ok_or(ReflectArchiveError::UnknownType(type_id))?;
                (
                    type_id,
                    prism_reflect::from_binary_with_limits(
                        blob,
                        types,
                        registration.type_info(),
                        limits,
                    )?,
                )
            }
        };
        if self.reflect.ops_for(id).is_none() {
            return Err(ReflectArchiveError::UnknownType(id));
        }
        Ok((id, decoded))
    }
}
