//! Bounded scene/entity/component identity mapping.
//!
//! This module is deliberately independent of a gameplay vocabulary.  A
//! caller supplies stable ids from its scene/persist layer (for example a
//! `PersistentId` as a `u128`) and a schema/component manifest.  The ECS
//! runtime handle is only a local lookup key; it is never treated as a wire
//! identity.
//!
//! The wire format is a bounded, deterministic snapshot:
//!
//! ```text
//! [magic: 4][wire_version: u8][schema_count: u32][entry_count: u32]
//! [schema: id u128, version u32] * schema_count
//! [entry: local u64, entity u128, schema u128, component u64,
//!         authority u128, revision u64] * entry_count
//! ```
//!
//! Schemas and entries are encoded in ascending key order.  Decoding rebuilds
//! the same bounded map and therefore rejects unknown schemas, identity aliasing,
//! duplicate keys, revision conflicts, and trailing bytes instead of silently
//! accepting a different mapping.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

const MAGIC: [u8; 4] = *b"PRMP";
/// The currently supported mapping snapshot wire version.
pub const WIRE_VERSION: u8 = 1;
#[cfg(test)]
const HEADER_LEN: usize = 13;
#[cfg(test)]
const SCHEMA_LEN: usize = 20;
#[cfg(test)]
const ENTRY_LEN: usize = 72;

/// Stable scene/entity identity carried over replication.
///
/// This width matches Prism's stable `PersistentId` contract without making
/// the replication crate depend on `prism_scene` or `prism_world_persist`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EntityId(u128);

impl EntityId {
    /// Wrap a stable raw identity.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Return the stable raw identity.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

/// A local ECS handle represented by `prism_ecs::Entity::to_bits()`.
///
/// The generation bits are retained so a despawn/reuse cannot accidentally
/// resolve to an older local entity.  This id is local-only and is not a
/// substitute for [`EntityId`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LocalEntityId(u64);

impl LocalEntityId {
    /// Construct from a packed ECS entity handle, rejecting generation zero.
    #[must_use]
    pub const fn from_entity_bits(bits: u64) -> Option<Self> {
        if (bits >> 32) == 0 {
            None
        } else {
            Some(Self(bits))
        }
    }

    /// Return the packed value accepted by `prism_ecs::Entity::from_bits`.
    #[must_use]
    pub const fn entity_bits(self) -> u64 {
        self.0
    }
}

/// Stable schema identity supplied by a scene/persist manifest.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SchemaId(u128);

impl SchemaId {
    /// Wrap a stable raw schema identity.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Return the raw schema identity.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

/// A stable component key within a schema.
///
/// `component` is intentionally supplied by a manifest.  A per-World dense
/// `prism_ecs::ComponentId` must not be used directly because it is not stable
/// across process/world construction order.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComponentKey {
    schema: SchemaId,
    component: u64,
}

impl ComponentKey {
    /// Create a key from a schema identity and manifest-assigned component id.
    #[must_use]
    pub const fn new(schema: SchemaId, component: u64) -> Self {
        Self { schema, component }
    }

    /// Return the owning schema identity.
    #[must_use]
    pub const fn schema(self) -> SchemaId {
        self.schema
    }

    /// Return the manifest-assigned component id.
    #[must_use]
    pub const fn component(self) -> u64 {
        self.component
    }
}

/// Stable authority/replica identity.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AuthorityId(u128);

impl AuthorityId {
    /// Wrap a stable raw authority identity.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Return the raw authority identity.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

/// Monotonic revision for one entity/component mapping.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Revision(u64);

impl Revision {
    /// Wrap a raw revision.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the raw revision.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A schema manifest entry required before component mappings can be inserted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchemaDescriptor {
    /// Stable schema identity.
    pub id: SchemaId,
    /// Schema ABI/field layout version.
    pub version: u32,
}

/// One local-to-wire entity/component mapping record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappingEntry {
    /// Local ECS handle bits used for lookup in this process.
    pub local: LocalEntityId,
    /// Stable scene/entity identity carried on the wire.
    pub entity: EntityId,
    /// Stable schema/component key.
    pub component: ComponentKey,
    /// Current authority for this replicated mapping.
    pub authority: AuthorityId,
    /// Monotonic update revision for this mapping.
    pub revision: Revision,
}

/// Bounds applied to schemas and live mapping entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappingLimits {
    /// Maximum number of registered schemas.
    pub max_schemas: usize,
    /// Maximum number of live entity/component entries.
    pub max_entries: usize,
}

impl MappingLimits {
    /// Create explicit bounds. Zero is valid and means that the corresponding
    /// collection is intentionally disabled.
    #[must_use]
    pub const fn new(max_schemas: usize, max_entries: usize) -> Self {
        Self {
            max_schemas,
            max_entries,
        }
    }
}

/// A structured mapping failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MappingError {
    /// The schema registry is full.
    SchemaCapacityExceeded {
        /// Maximum number of schemas allowed.
        max: usize,
    },
    /// The live mapping table is full.
    EntryCapacityExceeded {
        /// Maximum number of entries allowed.
        max: usize,
    },
    /// A schema was registered with a different version.
    SchemaConflict {
        /// Schema identity.
        id: SchemaId,
        /// Version already registered.
        expected_version: u32,
        /// Version supplied by the caller.
        found_version: u32,
    },
    /// An entry references a schema that has not been registered.
    UnknownSchema {
        /// Referenced schema identity.
        id: SchemaId,
    },
    /// A schema is repeated in one wire snapshot.
    DuplicateSchema {
        /// Repeated schema identity.
        id: SchemaId,
    },
    /// An entry is repeated in one wire snapshot.
    DuplicateEntry {
        /// Repeated stable entity identity.
        entity: EntityId,
        /// Repeated component key.
        component: ComponentKey,
    },
    /// One local ECS handle was assigned to two stable entity identities.
    LocalEntityCollision {
        /// Local ECS handle.
        local: LocalEntityId,
        /// Existing stable identity.
        existing: EntityId,
        /// Incoming stable identity.
        incoming: EntityId,
    },
    /// One stable entity/component key was assigned to two local handles.
    EntityCollision {
        /// Stable entity identity.
        entity: EntityId,
        /// Stable component key.
        component: ComponentKey,
        /// Existing local handle.
        existing: LocalEntityId,
        /// Incoming local handle.
        incoming: LocalEntityId,
    },
    /// An update arrived older than the retained mapping revision.
    StaleRevision {
        /// Stable entity identity.
        entity: EntityId,
        /// Stable component key.
        component: ComponentKey,
        /// Retained revision.
        current: Revision,
        /// Incoming revision.
        received: Revision,
    },
    /// The same revision arrived with different mapping contents.
    RevisionConflict {
        /// Stable entity identity.
        entity: EntityId,
        /// Stable component key.
        component: ComponentKey,
        /// Conflicting revision.
        revision: Revision,
    },
    /// The wire format version is not supported by this decoder.
    UnsupportedWireVersion {
        /// Version found in the payload.
        found: u8,
    },
    /// The four-byte wire marker is invalid.
    InvalidMagic {
        /// Marker found in the payload.
        found: [u8; 4],
    },
    /// A count in the payload exceeds the caller's configured bound.
    CountExceedsCapacity {
        /// True for schemas, false for entries.
        schemas: bool,
        /// Count found in the payload.
        count: u32,
        /// Configured maximum.
        max: usize,
    },
    /// A payload ended before a fixed-size field was available.
    Truncated {
        /// Byte offset where the missing field starts.
        offset: usize,
        /// Number of bytes required for the field.
        needed: usize,
    },
    /// An encoded local ECS handle has generation zero.
    InvalidLocalEntityBits {
        /// Invalid packed local handle.
        found: u64,
    },
    /// A payload contains bytes after the declared snapshot.
    TrailingBytes {
        /// Number of unconsumed bytes.
        remaining: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MappingKey {
    entity: EntityId,
    component: ComponentKey,
}

/// A bounded deterministic local-to-wire mapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplicationMapping {
    limits: MappingLimits,
    schemas: BTreeMap<SchemaId, SchemaDescriptor>,
    entries: BTreeMap<MappingKey, MappingEntry>,
    locals: BTreeMap<LocalEntityId, EntityId>,
}

impl ReplicationMapping {
    /// Create an empty mapping with explicit resource bounds.
    #[must_use]
    pub fn new(limits: MappingLimits) -> Self {
        Self {
            limits,
            schemas: BTreeMap::new(),
            entries: BTreeMap::new(),
            locals: BTreeMap::new(),
        }
    }

    /// Return the configured bounds.
    #[must_use]
    pub const fn limits(&self) -> MappingLimits {
        self.limits
    }

    /// Return the number of registered schemas.
    #[must_use]
    pub fn schema_len(&self) -> usize {
        self.schemas.len()
    }

    /// Return the number of live mapping entries.
    #[must_use]
    pub fn entry_len(&self) -> usize {
        self.entries.len()
    }

    /// Return the registered schema descriptor, if present.
    #[must_use]
    pub fn schema(&self, id: SchemaId) -> Option<SchemaDescriptor> {
        self.schemas.get(&id).copied()
    }

    /// Return the mapping for one stable entity/component key.
    #[must_use]
    pub fn get(&self, entity: EntityId, component: ComponentKey) -> Option<MappingEntry> {
        self.entries.get(&MappingKey { entity, component }).copied()
    }

    /// Resolve a local ECS handle to its stable entity identity.
    #[must_use]
    pub fn resolve_local(&self, local: LocalEntityId) -> Option<EntityId> {
        self.locals.get(&local).copied()
    }

    /// Resolve a stable entity identity to its current local ECS handle.
    ///
    /// The mapping table is component-keyed, so this walks the bounded entry
    /// set and returns the first deterministic match. A lifecycle provider can
    /// then use the returned local handle for an update or despawn.
    #[must_use]
    pub fn resolve_entity(&self, entity: EntityId) -> Option<LocalEntityId> {
        self.entries
            .values()
            .find(|entry| entry.entity == entity)
            .map(|entry| entry.local)
    }

    /// Remove every component mapping owned by one stable entity.
    ///
    /// Returns the number of removed component mappings. Removing an unknown
    /// entity is an idempotent no-op, which lets a retransmitted despawn be
    /// handled without rebinding a recycled ECS handle.
    pub fn remove_entity(&mut self, entity: EntityId) -> usize {
        let keys: Vec<_> = self
            .entries
            .keys()
            .copied()
            .filter(|key| key.entity == entity)
            .collect();
        let removed = keys.len();
        for key in keys {
            self.entries.remove(&key);
        }
        self.locals.retain(|_, owner| *owner != entity);
        removed
    }

    /// Remove one entity/component mapping, if present.
    pub fn remove_component(
        &mut self,
        entity: EntityId,
        component: ComponentKey,
    ) -> Option<MappingEntry> {
        let key = MappingKey { entity, component };
        let entry = self.entries.remove(&key)?;
        if !self
            .entries
            .values()
            .any(|candidate| candidate.local == entry.local)
        {
            self.locals.remove(&entry.local);
        }
        Some(entry)
    }

    /// Register a schema manifest entry. Re-registering the same version is
    /// idempotent; changing the version is rejected.
    pub fn register_schema(&mut self, schema: SchemaDescriptor) -> Result<(), MappingError> {
        if let Some(existing) = self.schemas.get(&schema.id) {
            if existing.version == schema.version {
                return Ok(());
            }
            return Err(MappingError::SchemaConflict {
                id: schema.id,
                expected_version: existing.version,
                found_version: schema.version,
            });
        }
        if self.schemas.len() >= self.limits.max_schemas {
            return Err(MappingError::SchemaCapacityExceeded {
                max: self.limits.max_schemas,
            });
        }
        self.schemas.insert(schema.id, schema);
        Ok(())
    }

    /// Insert or update one mapping.
    ///
    /// Updates must move strictly forward in revision. Repeating an identical
    /// record is idempotent. Identity bindings are never silently rebound:
    /// despawn/reuse or an authoritative migration must be represented by a
    /// higher-level lifecycle operation that removes the old mapping first.
    pub fn upsert(&mut self, entry: MappingEntry) -> Result<(), MappingError> {
        let schema = entry.component.schema();
        if !self.schemas.contains_key(&schema) {
            return Err(MappingError::UnknownSchema { id: schema });
        }

        let key = MappingKey {
            entity: entry.entity,
            component: entry.component,
        };
        if let Some(existing) = self.entries.get(&key).copied() {
            if existing == entry {
                return Ok(());
            }
            if entry.revision < existing.revision {
                return Err(MappingError::StaleRevision {
                    entity: entry.entity,
                    component: entry.component,
                    current: existing.revision,
                    received: entry.revision,
                });
            }
            if entry.revision == existing.revision {
                return Err(MappingError::RevisionConflict {
                    entity: entry.entity,
                    component: entry.component,
                    revision: entry.revision,
                });
            }
            if existing.local != entry.local {
                return Err(MappingError::EntityCollision {
                    entity: entry.entity,
                    component: entry.component,
                    existing: existing.local,
                    incoming: entry.local,
                });
            }
        }

        if let Some(existing) = self.locals.get(&entry.local).copied()
            && existing != entry.entity
        {
            return Err(MappingError::LocalEntityCollision {
                local: entry.local,
                existing,
                incoming: entry.entity,
            });
        }

        if !self.entries.contains_key(&key) && self.entries.len() >= self.limits.max_entries {
            return Err(MappingError::EntryCapacityExceeded {
                max: self.limits.max_entries,
            });
        }

        self.locals.insert(entry.local, entry.entity);
        self.entries.insert(key, entry);
        Ok(())
    }

    /// Return all schemas in deterministic ascending id order.
    pub fn schemas(&self) -> impl Iterator<Item = SchemaDescriptor> + '_ {
        self.schemas.values().copied()
    }

    /// Return all entries in deterministic `(entity, component)` order.
    pub fn entries(&self) -> impl Iterator<Item = MappingEntry> + '_ {
        self.entries.values().copied()
    }

    /// Encode the complete bounded mapping snapshot.
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MappingError> {
        let schema_count =
            u32::try_from(self.schemas.len()).map_err(|_| MappingError::CountExceedsCapacity {
                schemas: true,
                count: u32::MAX,
                max: self.limits.max_schemas,
            })?;
        let entry_count =
            u32::try_from(self.entries.len()).map_err(|_| MappingError::CountExceedsCapacity {
                schemas: false,
                count: u32::MAX,
                max: self.limits.max_entries,
            })?;

        out.extend_from_slice(&MAGIC);
        out.push(WIRE_VERSION);
        out.extend_from_slice(&schema_count.to_le_bytes());
        out.extend_from_slice(&entry_count.to_le_bytes());
        for schema in self.schemas.values() {
            out.extend_from_slice(&schema.id.as_u128().to_le_bytes());
            out.extend_from_slice(&schema.version.to_le_bytes());
        }
        for entry in self.entries.values() {
            out.extend_from_slice(&entry.local.entity_bits().to_le_bytes());
            out.extend_from_slice(&entry.entity.as_u128().to_le_bytes());
            out.extend_from_slice(&entry.component.schema().as_u128().to_le_bytes());
            out.extend_from_slice(&entry.component.component().to_le_bytes());
            out.extend_from_slice(&entry.authority.as_u128().to_le_bytes());
            out.extend_from_slice(&entry.revision.get().to_le_bytes());
        }
        Ok(())
    }

    /// Decode a complete mapping snapshot under explicit bounds.
    pub fn decode(bytes: &[u8], limits: MappingLimits) -> Result<Self, MappingError> {
        let mut cursor = Cursor::new(bytes);
        let magic = cursor.read_array::<4>()?;
        if magic != MAGIC {
            return Err(MappingError::InvalidMagic { found: magic });
        }
        let version = cursor.read_u8()?;
        if version != WIRE_VERSION {
            return Err(MappingError::UnsupportedWireVersion { found: version });
        }
        let schema_count = cursor.read_u32()?;
        if usize::try_from(schema_count).unwrap_or(usize::MAX) > limits.max_schemas {
            return Err(MappingError::CountExceedsCapacity {
                schemas: true,
                count: schema_count,
                max: limits.max_schemas,
            });
        }
        let entry_count = cursor.read_u32()?;
        if usize::try_from(entry_count).unwrap_or(usize::MAX) > limits.max_entries {
            return Err(MappingError::CountExceedsCapacity {
                schemas: false,
                count: entry_count,
                max: limits.max_entries,
            });
        }

        let mut mapping = Self::new(limits);
        for _ in 0..schema_count {
            let schema = SchemaDescriptor {
                id: SchemaId::from_u128(cursor.read_u128()?),
                version: cursor.read_u32()?,
            };
            if mapping.schema(schema.id).is_some() {
                return Err(MappingError::DuplicateSchema { id: schema.id });
            }
            mapping.register_schema(schema)?;
        }
        for _ in 0..entry_count {
            let local_bits = cursor.read_u64()?;
            let local = LocalEntityId::from_entity_bits(local_bits)
                .ok_or(MappingError::InvalidLocalEntityBits { found: local_bits })?;
            let entry = MappingEntry {
                local,
                entity: EntityId::from_u128(cursor.read_u128()?),
                component: ComponentKey::new(
                    SchemaId::from_u128(cursor.read_u128()?),
                    cursor.read_u64()?,
                ),
                authority: AuthorityId::from_u128(cursor.read_u128()?),
                revision: Revision::new(cursor.read_u64()?),
            };
            if mapping.get(entry.entity, entry.component).is_some() {
                return Err(MappingError::DuplicateEntry {
                    entity: entry.entity,
                    component: entry.component,
                });
            }
            mapping.upsert(entry)?;
        }
        if cursor.remaining() != 0 {
            return Err(MappingError::TrailingBytes {
                remaining: cursor.remaining(),
            });
        }
        Ok(mapping)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    const fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], MappingError> {
        if self.remaining() < N {
            return Err(MappingError::Truncated {
                offset: self.offset,
                needed: N,
            });
        }
        let mut out = [0; N];
        out.copy_from_slice(&self.bytes[self.offset..self.offset + N]);
        self.offset += N;
        Ok(out)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], MappingError> {
        self.take()
    }

    fn read_u8(&mut self) -> Result<u8, MappingError> {
        Ok(self.take::<1>()?[0])
    }

    fn read_u32(&mut self) -> Result<u32, MappingError> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }

    fn read_u64(&mut self) -> Result<u64, MappingError> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }

    fn read_u128(&mut self) -> Result<u128, MappingError> {
        Ok(u128::from_le_bytes(self.take::<16>()?))
    }
}

#[cfg(feature = "app")]
impl From<prism_ecs::entity::Entity> for LocalEntityId {
    fn from(entity: prism_ecs::entity::Entity) -> Self {
        Self(entity.to_bits())
    }
}

#[cfg(feature = "app")]
impl LocalEntityId {
    /// Convert back to a live-handle-shaped ECS entity, if its generation is
    /// valid. World liveness must still be checked by the caller.
    #[must_use]
    pub fn to_ecs(self) -> Option<prism_ecs::entity::Entity> {
        prism_ecs::entity::Entity::from_bits(self.entity_bits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(id: u128, version: u32) -> SchemaDescriptor {
        SchemaDescriptor {
            id: SchemaId::from_u128(id),
            version,
        }
    }

    fn entry(local: u64, entity: u128, component: u64, revision: u64) -> MappingEntry {
        MappingEntry {
            local: LocalEntityId::from_entity_bits(local).expect("valid generation"),
            entity: EntityId::from_u128(entity),
            component: ComponentKey::new(SchemaId::from_u128(7), component),
            authority: AuthorityId::from_u128(99),
            revision: Revision::new(revision),
        }
    }

    #[test]
    fn ecs_bits_preserve_generation_and_reject_zero() {
        assert!(LocalEntityId::from_entity_bits(3).is_none());
        let local = LocalEntityId::from_entity_bits((8_u64 << 32) | 4).unwrap();
        assert_eq!(local.entity_bits(), (8_u64 << 32) | 4);
    }

    #[cfg(feature = "app")]
    #[test]
    fn ecs_bridge_roundtrips_a_real_entity_handle() {
        let mut entities = prism_ecs::entity::Entities::new();
        let entity = entities.alloc();
        let local = LocalEntityId::from(entity);
        assert_eq!(local.to_ecs(), Some(entity));
    }

    #[test]
    fn mapping_is_bounded_and_revision_checked() {
        let mut mapping = ReplicationMapping::new(MappingLimits::new(1, 1));
        mapping.register_schema(schema(7, 3)).unwrap();
        mapping.upsert(entry((1_u64 << 32) | 1, 101, 4, 1)).unwrap();
        assert_eq!(
            mapping.upsert(entry((1_u64 << 32) | 1, 101, 4, 0)),
            Err(MappingError::StaleRevision {
                entity: EntityId::from_u128(101),
                component: ComponentKey::new(SchemaId::from_u128(7), 4),
                current: Revision::new(1),
                received: Revision::new(0),
            })
        );
        assert_eq!(
            mapping.upsert(entry((1_u64 << 32) | 1, 102, 5, 1)),
            Err(MappingError::LocalEntityCollision {
                local: LocalEntityId::from_entity_bits((1_u64 << 32) | 1).unwrap(),
                existing: EntityId::from_u128(101),
                incoming: EntityId::from_u128(102),
            })
        );
        assert_eq!(mapping.entry_len(), 1);
        assert_eq!(
            mapping.register_schema(schema(8, 1)),
            Err(MappingError::SchemaCapacityExceeded { max: 1 })
        );
    }

    #[test]
    fn encoding_is_deterministic_and_roundtrips() {
        let limits = MappingLimits::new(2, 2);
        let mut first = ReplicationMapping::new(limits);
        first.register_schema(schema(9, 1)).unwrap();
        first.register_schema(schema(7, 3)).unwrap();
        first.upsert(entry((2_u64 << 32) | 2, 202, 8, 2)).unwrap();
        first.upsert(entry((1_u64 << 32) | 1, 101, 4, 1)).unwrap();

        let mut second = ReplicationMapping::new(limits);
        second.register_schema(schema(7, 3)).unwrap();
        second.register_schema(schema(9, 1)).unwrap();
        second.upsert(entry((1_u64 << 32) | 1, 101, 4, 1)).unwrap();
        second.upsert(entry((2_u64 << 32) | 2, 202, 8, 2)).unwrap();

        let mut a = Vec::new();
        let mut b = Vec::new();
        first.encode(&mut a).unwrap();
        second.encode(&mut b).unwrap();
        assert_eq!(a, b);
        assert_eq!(ReplicationMapping::decode(&a, limits).unwrap(), first);
    }

    #[test]
    fn malformed_version_capacity_and_trailing_bytes_are_structured() {
        let mut mapping = ReplicationMapping::new(MappingLimits::new(1, 1));
        mapping.register_schema(schema(7, 3)).unwrap();
        mapping.upsert(entry((1_u64 << 32) | 1, 101, 4, 1)).unwrap();
        let mut bytes = Vec::new();
        mapping.encode(&mut bytes).unwrap();

        bytes[4] = WIRE_VERSION + 1;
        assert_eq!(
            ReplicationMapping::decode(&bytes, MappingLimits::new(1, 1)),
            Err(MappingError::UnsupportedWireVersion {
                found: WIRE_VERSION + 1
            })
        );

        bytes.clear();
        mapping.encode(&mut bytes).unwrap();
        bytes[9..13].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(
            ReplicationMapping::decode(&bytes, MappingLimits::new(1, 1)),
            Err(MappingError::CountExceedsCapacity {
                schemas: false,
                count: 2,
                max: 1,
            })
        );

        let mut valid = Vec::new();
        mapping.encode(&mut valid).unwrap();
        valid.push(0);
        assert_eq!(
            ReplicationMapping::decode(&valid, MappingLimits::new(1, 1)),
            Err(MappingError::TrailingBytes { remaining: 1 })
        );
    }

    #[test]
    fn constants_match_wire_layout() {
        assert_eq!(HEADER_LEN + SCHEMA_LEN + ENTRY_LEN, 105);
    }
}
