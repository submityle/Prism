//! Deterministic compilation from explicit scene authoring records to runtime templates.
//!
//! The baker accepts canonical component rows and schema-declared entity-reference
//! offsets. It never guesses Rust layouts or reinterprets opaque bytes. Source import,
//! reflection-driven field serialization, and CAS/package publication are separate
//! adapters; the output here is the validated `prism_scene::SceneTemplate` contract.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use prism_asset::StableGuid;
use prism_scene::overrides::{ArchetypeSignature, ComponentTypeId};
use prism_scene::{
    ArchetypeGroup, Column, FieldSlot, InstantiationPlan, LocalEntityId, NameTable, NestedSlot,
    ResidencyPlan, ResidencyTier, SceneTemplate,
};

const SOURCE_MAGIC: &[u8; 4] = b"PRSA";
const SOURCE_VERSION: u32 = 1;

/// Editable payload kind for PRSA scene source documents.
pub const SCENE_SOURCE_KIND: u128 = 0x5052_4953_4d5f_5343_454e_455f_5352_4331;
/// Runtime artifact kind emitted for validated PRST scene templates.
pub const SCENE_TEMPLATE_KIND: u128 = prism_scene::SCENE_TEMPLATE_KIND;
/// Current scene content schema emitted by the standard scene cooker.
pub const SCENE_TEMPLATE_SCHEMA: u32 = prism_scene::SCENE_TEMPLATE_SCHEMA;

/// A component's fixed canonical row width and explicit local-entity reference offsets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentSchema {
    /// Serialized bytes per entity for this component.
    pub row_bytes: u32,
    /// Byte offsets containing little-endian `LocalEntityId` values.
    pub entity_reference_offsets: Vec<u32>,
}

/// One component row in authored canonical wire form.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceComponent {
    /// Stable component type identity.
    pub component: ComponentTypeId,
    /// Exactly one schema-sized row; references are encoded as little-endian `u32`.
    pub row: Vec<u8>,
}

/// One authored scene entity with a stable template-local identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceEntity {
    /// Dense base-segment local ID. IDs must cover `0..entities.len()`.
    pub local: LocalEntityId,
    /// Optional unique authored name.
    pub name: Option<String>,
    /// Parent local ID; local zero is the sole root and has no parent.
    pub parent: Option<LocalEntityId>,
    /// Residency annotation; Interactive is represented by the plan default.
    pub tier: ResidencyTier,
    /// Components in any input order; the baker canonicalizes by component ID.
    pub components: Vec<SourceComponent>,
}

/// In-memory authoring document consumed by [`bake_scene`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneSource {
    /// Stable identity of the resulting runtime template.
    pub guid: StableGuid,
    /// Entity records. Input order is not semantically significant.
    pub entities: Vec<SourceEntity>,
    /// Schemas required to interpret all component rows.
    pub schemas: BTreeMap<ComponentTypeId, ComponentSchema>,
    /// Nested templates rooted at base entities.
    pub nested: Vec<NestedSlot>,
    /// Canonically sorted transitive dependency closure.
    pub dependencies: Vec<StableGuid>,
}

/// Explicit limits applied before output buffers are allocated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BakeLimits {
    /// Maximum authored entity count.
    pub max_entities: usize,
    /// Maximum total component rows.
    pub max_component_rows: usize,
    /// Maximum aggregate baked column bytes.
    pub max_column_bytes: usize,
    /// Maximum aggregate UTF-8 bytes in authored names.
    pub max_name_bytes: usize,
    /// Maximum nested template slots.
    pub max_nested: usize,
    /// Maximum dependency GUIDs.
    pub max_dependencies: usize,
    /// Maximum entity-reference slots emitted into the plan.
    pub max_entity_reference_slots: usize,
}

impl Default for BakeLimits {
    fn default() -> Self {
        Self {
            max_entities: 1_048_576,
            max_component_rows: 16_777_216,
            max_column_bytes: 1 << 30,
            max_name_bytes: 16 * 1024 * 1024,
            max_nested: 1_048_576,
            max_dependencies: 1_048_576,
            max_entity_reference_slots: 16_777_216,
        }
    }
}

/// Deterministic validation or resource failure while compiling a scene source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BakeError {
    /// The source template identity is nil.
    NilGuid,
    /// A configured entity, row, or byte limit was exceeded.
    LimitExceeded(&'static str),
    /// An entity local ID is reserved, outside the dense range, or duplicated.
    InvalidLocal(LocalEntityId),
    /// IDs do not form a dense `0..entity_count` range.
    MissingLocal(LocalEntityId),
    /// An entity component has no registered schema.
    MissingSchema(ComponentTypeId),
    /// A component row has a different width than the registered schema.
    RowWidth {
        /// Entity owning the malformed row.
        local: LocalEntityId,
        /// Component whose row failed validation.
        component: ComponentTypeId,
        /// Expected fixed width.
        expected: u32,
        /// Actual row width.
        actual: usize,
    },
    /// An entity declares a component more than once.
    DuplicateComponent {
        /// Entity owning the duplicate.
        local: LocalEntityId,
        /// Repeated component identity.
        component: ComponentTypeId,
    },
    /// A schema's entity-reference slots overlap, are unsorted, or exceed its row.
    InvalidReferenceLayout {
        /// Component whose schema is invalid.
        component: ComponentTypeId,
        /// Invalid byte offset.
        offset: u32,
    },
    /// An encoded entity reference names no source entity.
    UnknownReference {
        /// Entity row containing the reference.
        owner: LocalEntityId,
        /// Referenced local identity.
        target: LocalEntityId,
    },
    /// A named entity name is empty or duplicated.
    InvalidName(LocalEntityId),
    /// An authoring graph violates the runtime plan hierarchy contract.
    InvalidPlan(prism_scene::PlanError),
    /// Runtime template validation rejected nested, dependency, or tier metadata.
    InvalidTemplate(prism_scene::TemplateError),
}

/// Compiles explicit source records to canonical archetype columns and a validated template.
pub fn bake_scene(source: &SceneSource, limits: BakeLimits) -> Result<SceneTemplate, BakeError> {
    if source.guid.is_nil() {
        return Err(BakeError::NilGuid);
    }
    if source.entities.len() > limits.max_entities {
        return Err(BakeError::LimitExceeded("entities"));
    }
    if source.nested.len() > limits.max_nested {
        return Err(BakeError::LimitExceeded("nested slots"));
    }
    if source.dependencies.len() > limits.max_dependencies {
        return Err(BakeError::LimitExceeded("dependencies"));
    }
    let entity_count = u32::try_from(source.entities.len())
        .map_err(|_| BakeError::LimitExceeded("entity id range"))?;

    let mut entities = BTreeMap::new();
    for entity in &source.entities {
        if entity.local.is_reserved() || entity.local.index() >= entity_count {
            return Err(BakeError::InvalidLocal(entity.local));
        }
        if entities.insert(entity.local, entity).is_some() {
            return Err(BakeError::InvalidLocal(entity.local));
        }
    }
    for index in 0..entity_count {
        let local = LocalEntityId::new(index);
        if !entities.contains_key(&local) {
            return Err(BakeError::MissingLocal(local));
        }
    }

    let mut row_count = 0usize;
    let mut column_bytes = 0usize;
    let mut name_bytes = 0usize;
    let mut reference_slots = 0usize;
    let mut groups: BTreeMap<Vec<ComponentTypeId>, Vec<(LocalEntityId, Vec<&SourceComponent>)>> =
        BTreeMap::new();
    let mut names = NameTable::new();
    let mut tiers = ResidencyPlan::new();
    let mut hierarchy = Vec::new();

    for entity in entities.values().copied() {
        if let Some(name) = &entity.name {
            name_bytes = name_bytes
                .checked_add(name.len())
                .ok_or(BakeError::LimitExceeded("name bytes"))?;
            if name_bytes > limits.max_name_bytes {
                return Err(BakeError::LimitExceeded("name bytes"));
            }
            if name.is_empty() || names.insert(name.clone(), entity.local).is_some() {
                return Err(BakeError::InvalidName(entity.local));
            }
        }
        if entity.tier != ResidencyTier::Interactive {
            tiers.set(entity.local, entity.tier);
        }
        if let Some(parent) = entity.parent {
            hierarchy.push((entity.local, parent));
        }

        let mut component_ids = Vec::new();
        let mut component_rows = Vec::with_capacity(entity.components.len());
        let mut seen = BTreeSet::new();
        for component in &entity.components {
            if !seen.insert(component.component) {
                return Err(BakeError::DuplicateComponent {
                    local: entity.local,
                    component: component.component,
                });
            }
            let schema = source
                .schemas
                .get(&component.component)
                .ok_or(BakeError::MissingSchema(component.component))?;
            validate_reference_layout(component.component, schema)?;
            reference_slots = reference_slots
                .checked_add(schema.entity_reference_offsets.len())
                .ok_or(BakeError::LimitExceeded("entity reference slots"))?;
            if reference_slots > limits.max_entity_reference_slots {
                return Err(BakeError::LimitExceeded("entity reference slots"));
            }
            if component.row.len() != schema.row_bytes as usize {
                return Err(BakeError::RowWidth {
                    local: entity.local,
                    component: component.component,
                    expected: schema.row_bytes,
                    actual: component.row.len(),
                });
            }
            for &offset in &schema.entity_reference_offsets {
                let start = offset as usize;
                let end = start + 4;
                let target = LocalEntityId::new(u32::from_le_bytes(
                    component.row[start..end]
                        .try_into()
                        .expect("validated reference width"),
                ));
                if !entities.contains_key(&target) {
                    return Err(BakeError::UnknownReference {
                        owner: entity.local,
                        target,
                    });
                }
            }
            row_count = row_count
                .checked_add(1)
                .ok_or(BakeError::LimitExceeded("component rows"))?;
            column_bytes = column_bytes
                .checked_add(component.row.len())
                .ok_or(BakeError::LimitExceeded("column bytes"))?;
            if row_count > limits.max_component_rows {
                return Err(BakeError::LimitExceeded("component rows"));
            }
            if column_bytes > limits.max_column_bytes {
                return Err(BakeError::LimitExceeded("column bytes"));
            }
            component_ids.push(component.component);
            component_rows.push(component);
        }
        let mut ordered: Vec<_> = component_ids.into_iter().zip(component_rows).collect();
        ordered.sort_unstable_by_key(|(component, _)| *component);
        let (component_ids, component_rows): (Vec<_>, Vec<_>) = ordered.into_iter().unzip();
        groups
            .entry(component_ids)
            .or_default()
            .push((entity.local, component_rows));
    }

    let mut archetype_groups = Vec::new();
    let mut entity_valued_fields = Vec::new();
    for (group_index, (signature_ids, mut group_entities)) in groups.into_iter().enumerate() {
        let group_index =
            u32::try_from(group_index).map_err(|_| BakeError::LimitExceeded("archetype groups"))?;
        group_entities.sort_unstable_by_key(|(local, _)| *local);
        let signature = ArchetypeSignature::new(signature_ids.clone());
        let locals = group_entities
            .iter()
            .map(|(local, _)| *local)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut columns = Vec::new();
        for (column_index, component_id) in signature_ids.iter().copied().enumerate() {
            let column_index = u32::try_from(column_index)
                .map_err(|_| BakeError::LimitExceeded("columns per archetype"))?;
            let schema = &source.schemas[&component_id];
            let group_entity_count = group_entities.len();
            let group_bytes = (schema.row_bytes as usize)
                .checked_mul(group_entity_count)
                .ok_or(BakeError::LimitExceeded("column bytes"))?;
            let mut blob = Vec::new();
            blob.try_reserve_exact(group_bytes)
                .map_err(|_| BakeError::LimitExceeded("column allocation"))?;
            for (row_index, (local, component_rows)) in group_entities.iter().enumerate() {
                let row = component_rows
                    .get(column_index as usize)
                    .expect("ordered component rows match the archetype signature");
                debug_assert_eq!(row.component, component_id);
                blob.extend_from_slice(&row.row);
                for &byte_offset in &schema.entity_reference_offsets {
                    entity_valued_fields.push(FieldSlot {
                        local: *local,
                        group: group_index,
                        column: column_index,
                        byte_offset,
                    });
                }
                debug_assert_eq!(blob.len(), schema.row_bytes as usize * (row_index + 1),);
            }
            columns.push(Column::new(component_id, Arc::<[u8]>::from(blob)));
        }
        archetype_groups.push(ArchetypeGroup::new(
            signature,
            locals,
            columns.into_boxed_slice(),
        ));
    }

    entity_valued_fields
        .sort_unstable_by_key(|slot| (slot.local, slot.group, slot.column, slot.byte_offset));
    let plan = InstantiationPlan {
        entity_count,
        groups: archetype_groups.into_boxed_slice(),
        entity_valued_fields: entity_valued_fields.into_boxed_slice(),
        hierarchy: hierarchy.into_boxed_slice(),
        reserved_local_base: LocalEntityId::RESERVED_BASE,
    };
    plan.validate().map_err(BakeError::InvalidPlan)?;

    let mut dependencies = source.dependencies.clone();
    dependencies.sort_unstable();
    dependencies.dedup();
    let template = SceneTemplate {
        guid: source.guid,
        schema: SCENE_TEMPLATE_SCHEMA,
        plan,
        names,
        nested: source.nested.clone().into_boxed_slice(),
        tiers,
        deps: dependencies.into_boxed_slice(),
    };
    template.validate().map_err(BakeError::InvalidTemplate)?;
    Ok(template)
}

/// Compiles and encodes a validated template using the runtime PRST wire contract.
pub fn bake_scene_bytes(
    source: &SceneSource,
    limits: BakeLimits,
) -> Result<Vec<u8>, BakeOutputError> {
    let template = bake_scene(source, limits).map_err(BakeOutputError::Bake)?;
    prism_scene::encode_template(&template).map_err(BakeOutputError::Codec)
}

/// Failure compiling or encoding a baked scene artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BakeOutputError {
    /// Source data failed semantic compilation.
    Bake(BakeError),
    /// The runtime template codec rejected its output contract.
    Codec(prism_scene::TemplateCodecError),
}

/// Bounds applied to persisted scene authoring payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SceneSourceCodecLimits {
    /// Maximum complete PRSA payload size.
    pub max_bytes: usize,
    /// Semantic count and allocation limits forwarded to the baker.
    pub bake: BakeLimits,
}

impl Default for SceneSourceCodecLimits {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            bake: BakeLimits::default(),
        }
    }
}

/// Invalid or over-budget scene authoring payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SceneSourceCodecError {
    /// Payload has invalid magic, version, encoding or trailing bytes.
    Invalid(&'static str),
    /// A bounded output or allocation could not be represented.
    LimitExceeded(&'static str),
    /// Source rows failed semantic validation.
    Bake(BakeError),
}

/// Encodes a validated source document as a deterministic PRSA v1 payload.
pub fn encode_scene_source(
    source: &SceneSource,
    limits: SceneSourceCodecLimits,
) -> Result<Vec<u8>, SceneSourceCodecError> {
    bake_scene(source, limits.bake).map_err(SceneSourceCodecError::Bake)?;
    let mut writer = SourceWriter::new(limits.max_bytes);
    writer.bytes(SOURCE_MAGIC)?;
    writer.u32(SOURCE_VERSION)?;
    writer.u128(source.guid.to_u128())?;
    writer.count(
        source.schemas.len(),
        limits.bake.max_component_rows,
        "schemas",
    )?;
    for (component, schema) in &source.schemas {
        writer.u64(component.raw())?;
        writer.u32(schema.row_bytes)?;
        writer.count(
            schema.entity_reference_offsets.len(),
            limits.bake.max_entity_reference_slots,
            "schema references",
        )?;
        for offset in &schema.entity_reference_offsets {
            writer.u32(*offset)?;
        }
    }

    let mut entities = source.entities.iter().collect::<Vec<_>>();
    entities.sort_unstable_by_key(|entity| entity.local);
    writer.count(entities.len(), limits.bake.max_entities, "entities")?;
    let mut aggregate_name_bytes = 0usize;
    let mut rows = 0usize;
    let mut row_bytes = 0usize;
    for entity in entities {
        writer.u32(entity.local.index())?;
        match &entity.name {
            Some(name) => {
                aggregate_name_bytes = aggregate_name_bytes
                    .checked_add(name.len())
                    .ok_or(SceneSourceCodecError::LimitExceeded("name bytes"))?;
                if aggregate_name_bytes > limits.bake.max_name_bytes {
                    return Err(SceneSourceCodecError::LimitExceeded("name bytes"));
                }
                writer.u8(1)?;
                writer.sized_bytes(name.as_bytes(), u32::MAX as usize, "name")?;
            }
            None => writer.u8(0)?,
        }
        match entity.parent {
            Some(parent) => {
                writer.u8(1)?;
                writer.u32(parent.index())?;
            }
            None => writer.u8(0)?,
        }
        writer.u8(match entity.tier {
            ResidencyTier::Interactive => 0,
            ResidencyTier::Streamed => 1,
            ResidencyTier::Proxy => 2,
        })?;
        writer.count(
            entity.components.len(),
            limits.bake.max_component_rows,
            "entity components",
        )?;
        let mut components = entity.components.iter().collect::<Vec<_>>();
        components.sort_unstable_by_key(|component| component.component);
        for component in components {
            rows = rows
                .checked_add(1)
                .ok_or(SceneSourceCodecError::LimitExceeded("component rows"))?;
            row_bytes = row_bytes
                .checked_add(component.row.len())
                .ok_or(SceneSourceCodecError::LimitExceeded("column bytes"))?;
            if rows > limits.bake.max_component_rows || row_bytes > limits.bake.max_column_bytes {
                return Err(SceneSourceCodecError::LimitExceeded("component data"));
            }
            writer.u64(component.component.raw())?;
            writer.sized_bytes(&component.row, u32::MAX as usize, "component row")?;
        }
    }
    writer.count(source.nested.len(), limits.bake.max_nested, "nested slots")?;
    for slot in &source.nested {
        writer.u32(slot.root.index())?;
        writer.u128(slot.template.to_u128())?;
    }
    writer.count(
        source.dependencies.len(),
        limits.bake.max_dependencies,
        "dependencies",
    )?;
    let mut dependencies = source.dependencies.clone();
    dependencies.sort_unstable();
    dependencies.dedup();
    for dependency in &dependencies {
        writer.u128(dependency.to_u128())?;
    }
    Ok(writer.finish())
}

/// Decodes PRSA v1 under explicit byte, count and allocation budgets.
///
/// The decoded records must still pass [`bake_scene`] before they are trusted as
/// runtime scene content.
pub fn decode_scene_source(
    bytes: &[u8],
    limits: SceneSourceCodecLimits,
) -> Result<SceneSource, SceneSourceCodecError> {
    if bytes.len() > limits.max_bytes {
        return Err(SceneSourceCodecError::LimitExceeded("encoded bytes"));
    }
    let mut reader = SourceReader { bytes, offset: 0 };
    if reader.take(4)? != SOURCE_MAGIC {
        return Err(SceneSourceCodecError::Invalid("magic"));
    }
    if reader.u32()? != SOURCE_VERSION {
        return Err(SceneSourceCodecError::Invalid("version"));
    }
    let guid = StableGuid::from_u128(reader.u128()?);
    let schema_count = reader.count(limits.bake.max_component_rows, "schemas", 16)?;
    let mut schemas = BTreeMap::new();
    let mut aggregate_schema_reference_slots = 0usize;
    for _ in 0..schema_count {
        let component = ComponentTypeId::from_raw(reader.u64()?);
        let row_bytes = reader.u32()?;
        let reference_count = reader.count(
            limits.bake.max_entity_reference_slots,
            "schema references",
            4,
        )?;
        aggregate_schema_reference_slots = aggregate_schema_reference_slots
            .checked_add(reference_count)
            .ok_or(SceneSourceCodecError::LimitExceeded("schema references"))?;
        if aggregate_schema_reference_slots > limits.bake.max_entity_reference_slots {
            return Err(SceneSourceCodecError::LimitExceeded("schema references"));
        }
        let mut entity_reference_offsets = Vec::new();
        entity_reference_offsets
            .try_reserve_exact(reference_count)
            .map_err(|_| SceneSourceCodecError::LimitExceeded("schema references allocation"))?;
        for _ in 0..reference_count {
            entity_reference_offsets.push(reader.u32()?);
        }
        if schemas
            .insert(
                component,
                ComponentSchema {
                    row_bytes,
                    entity_reference_offsets,
                },
            )
            .is_some()
        {
            return Err(SceneSourceCodecError::Invalid("duplicate schema"));
        }
    }

    let entity_count = reader.count(limits.bake.max_entities, "entities", 11)?;
    let mut entities = Vec::new();
    entities
        .try_reserve_exact(entity_count)
        .map_err(|_| SceneSourceCodecError::LimitExceeded("entities allocation"))?;
    let mut aggregate_name_bytes = 0usize;
    let mut aggregate_rows = 0usize;
    let mut aggregate_column_bytes = 0usize;
    let mut aggregate_reference_slots = 0usize;
    for _ in 0..entity_count {
        let local = LocalEntityId::new(reader.u32()?);
        let name = match reader.u8()? {
            0 => None,
            1 => {
                let bytes = reader.sized_bytes(limits.bake.max_name_bytes, "name")?;
                aggregate_name_bytes = aggregate_name_bytes
                    .checked_add(bytes.len())
                    .ok_or(SceneSourceCodecError::LimitExceeded("name bytes"))?;
                if aggregate_name_bytes > limits.bake.max_name_bytes {
                    return Err(SceneSourceCodecError::LimitExceeded("name bytes"));
                }
                let text = core::str::from_utf8(bytes)
                    .map_err(|_| SceneSourceCodecError::Invalid("name utf8"))?;
                let mut name = String::new();
                name.try_reserve_exact(text.len())
                    .map_err(|_| SceneSourceCodecError::LimitExceeded("name allocation"))?;
                name.push_str(text);
                Some(name)
            }
            _ => return Err(SceneSourceCodecError::Invalid("name tag")),
        };
        let parent = match reader.u8()? {
            0 => None,
            1 => Some(LocalEntityId::new(reader.u32()?)),
            _ => return Err(SceneSourceCodecError::Invalid("parent tag")),
        };
        let tier = match reader.u8()? {
            0 => ResidencyTier::Interactive,
            1 => ResidencyTier::Streamed,
            2 => ResidencyTier::Proxy,
            _ => return Err(SceneSourceCodecError::Invalid("residency tier")),
        };
        let component_count =
            reader.count(limits.bake.max_component_rows, "entity components", 12)?;
        aggregate_rows = aggregate_rows
            .checked_add(component_count)
            .ok_or(SceneSourceCodecError::LimitExceeded("component rows"))?;
        if aggregate_rows > limits.bake.max_component_rows {
            return Err(SceneSourceCodecError::LimitExceeded("component rows"));
        }
        let mut components = Vec::new();
        components
            .try_reserve_exact(component_count)
            .map_err(|_| SceneSourceCodecError::LimitExceeded("components allocation"))?;
        for _ in 0..component_count {
            let component = ComponentTypeId::from_raw(reader.u64()?);
            let row = reader.sized_bytes(limits.bake.max_column_bytes, "component row")?;
            aggregate_column_bytes = aggregate_column_bytes
                .checked_add(row.len())
                .ok_or(SceneSourceCodecError::LimitExceeded("column bytes"))?;
            if aggregate_column_bytes > limits.bake.max_column_bytes {
                return Err(SceneSourceCodecError::LimitExceeded("column bytes"));
            }
            let schema = schemas
                .get(&component)
                .ok_or(SceneSourceCodecError::Invalid("component schema missing"))?;
            aggregate_reference_slots = aggregate_reference_slots
                .checked_add(schema.entity_reference_offsets.len())
                .ok_or(SceneSourceCodecError::LimitExceeded(
                    "entity reference slots",
                ))?;
            if aggregate_reference_slots > limits.bake.max_entity_reference_slots {
                return Err(SceneSourceCodecError::LimitExceeded(
                    "entity reference slots",
                ));
            }
            let mut owned_row = Vec::new();
            owned_row
                .try_reserve_exact(row.len())
                .map_err(|_| SceneSourceCodecError::LimitExceeded("component allocation"))?;
            owned_row.extend_from_slice(row);
            components.push(SourceComponent {
                component,
                row: owned_row,
            });
        }
        entities.push(SourceEntity {
            local,
            name,
            parent,
            tier,
            components,
        });
    }
    let nested_count = reader.count(limits.bake.max_nested, "nested slots", 20)?;
    let mut nested = Vec::new();
    nested
        .try_reserve_exact(nested_count)
        .map_err(|_| SceneSourceCodecError::LimitExceeded("nested allocation"))?;
    for _ in 0..nested_count {
        nested.push(NestedSlot::new(
            LocalEntityId::new(reader.u32()?),
            StableGuid::from_u128(reader.u128()?),
        ));
    }
    let dependency_count = reader.count(limits.bake.max_dependencies, "dependencies", 16)?;
    let mut dependencies = Vec::new();
    dependencies
        .try_reserve_exact(dependency_count)
        .map_err(|_| SceneSourceCodecError::LimitExceeded("dependencies allocation"))?;
    for _ in 0..dependency_count {
        dependencies.push(StableGuid::from_u128(reader.u128()?));
    }
    if reader.offset != bytes.len() {
        return Err(SceneSourceCodecError::Invalid("trailing bytes"));
    }
    let source = SceneSource {
        guid,
        entities,
        schemas,
        nested,
        dependencies,
    };
    Ok(source)
}

struct SourceWriter {
    bytes: Vec<u8>,
    max_bytes: usize,
}

impl SourceWriter {
    fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            max_bytes,
        }
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), SceneSourceCodecError> {
        let new_len = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or(SceneSourceCodecError::LimitExceeded("encoded bytes"))?;
        if new_len > self.max_bytes {
            return Err(SceneSourceCodecError::LimitExceeded("encoded bytes"));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|_| SceneSourceCodecError::LimitExceeded("encoded allocation"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), SceneSourceCodecError> {
        self.bytes(&[value])
    }
    fn u32(&mut self, value: u32) -> Result<(), SceneSourceCodecError> {
        self.bytes(&value.to_le_bytes())
    }
    fn u64(&mut self, value: u64) -> Result<(), SceneSourceCodecError> {
        self.bytes(&value.to_le_bytes())
    }
    fn u128(&mut self, value: u128) -> Result<(), SceneSourceCodecError> {
        self.bytes(&value.to_le_bytes())
    }
    fn count(
        &mut self,
        value: usize,
        limit: usize,
        label: &'static str,
    ) -> Result<(), SceneSourceCodecError> {
        if value > limit {
            return Err(SceneSourceCodecError::LimitExceeded(label));
        }
        self.u32(u32::try_from(value).map_err(|_| SceneSourceCodecError::LimitExceeded(label))?)
    }
    fn sized_bytes(
        &mut self,
        value: &[u8],
        limit: usize,
        label: &'static str,
    ) -> Result<(), SceneSourceCodecError> {
        if value.len() > limit {
            return Err(SceneSourceCodecError::LimitExceeded(label));
        }
        self.u32(
            u32::try_from(value.len()).map_err(|_| SceneSourceCodecError::LimitExceeded(label))?,
        )?;
        self.bytes(value)
    }
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

struct SourceReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SourceReader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], SceneSourceCodecError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(SceneSourceCodecError::Invalid("length overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(SceneSourceCodecError::Invalid("truncated payload"))?;
        self.offset = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, SceneSourceCodecError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, SceneSourceCodecError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed width"),
        ))
    }
    fn u64(&mut self) -> Result<u64, SceneSourceCodecError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed width"),
        ))
    }
    fn u128(&mut self) -> Result<u128, SceneSourceCodecError> {
        Ok(u128::from_le_bytes(
            self.take(16)?.try_into().expect("fixed width"),
        ))
    }
    fn count(
        &mut self,
        limit: usize,
        label: &'static str,
        minimum_item_bytes: usize,
    ) -> Result<usize, SceneSourceCodecError> {
        let count = self.u32()? as usize;
        if count > limit {
            return Err(SceneSourceCodecError::LimitExceeded(label));
        }
        if count > self.bytes.len().saturating_sub(self.offset) / minimum_item_bytes {
            return Err(SceneSourceCodecError::Invalid("truncated count payload"));
        }
        Ok(count)
    }
    fn sized_bytes(
        &mut self,
        limit: usize,
        label: &'static str,
    ) -> Result<&'a [u8], SceneSourceCodecError> {
        let len = self.u32()? as usize;
        if len > limit {
            return Err(SceneSourceCodecError::LimitExceeded(label));
        }
        self.take(len)
    }
}

fn validate_reference_layout(
    component: ComponentTypeId,
    schema: &ComponentSchema,
) -> Result<(), BakeError> {
    let mut previous_end = 0u32;
    for &offset in &schema.entity_reference_offsets {
        let end = offset.checked_add(4);
        if offset < previous_end || end.is_none_or(|end| end > schema.row_bytes) {
            return Err(BakeError::InvalidReferenceLayout { component, offset });
        }
        previous_end = end.expect("checked above");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_scene::{decode_template, TemplateDecodeLimits};

    fn source() -> SceneSource {
        let link = ComponentTypeId::from_raw(7);
        let marker = ComponentTypeId::from_raw(3);
        let mut schemas = BTreeMap::new();
        schemas.insert(
            link,
            ComponentSchema {
                row_bytes: 4,
                entity_reference_offsets: vec![0],
            },
        );
        schemas.insert(
            marker,
            ComponentSchema {
                row_bytes: 4,
                entity_reference_offsets: vec![],
            },
        );
        SceneSource {
            guid: StableGuid::from_u128(100),
            entities: vec![
                SourceEntity {
                    local: LocalEntityId::new(1),
                    name: Some("child".to_owned()),
                    parent: Some(LocalEntityId::new(0)),
                    tier: ResidencyTier::Proxy,
                    components: vec![
                        SourceComponent {
                            component: link,
                            row: 0u32.to_le_bytes().to_vec(),
                        },
                        SourceComponent {
                            component: marker,
                            row: 9u32.to_le_bytes().to_vec(),
                        },
                    ],
                },
                SourceEntity {
                    local: LocalEntityId::new(0),
                    name: Some("root".to_owned()),
                    parent: None,
                    tier: ResidencyTier::Interactive,
                    components: vec![SourceComponent {
                        component: marker,
                        row: 1u32.to_le_bytes().to_vec(),
                    }],
                },
            ],
            schemas,
            nested: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    #[test]
    fn bakes_canonical_groups_and_round_trips_runtime_template() {
        let source = source();
        let first = bake_scene(&source, BakeLimits::default()).unwrap();
        let second = bake_scene(&source, BakeLimits::default()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.plan.groups.len(), 2);
        assert_eq!(first.plan.entity_valued_fields.len(), 1);
        assert_eq!(first.names.get("child"), Some(LocalEntityId::new(1)));
        assert_eq!(
            first.tiers.tier_of(LocalEntityId::new(1)),
            ResidencyTier::Proxy
        );
        let bytes = bake_scene_bytes(&source, BakeLimits::default()).unwrap();
        assert_eq!(
            decode_template(&bytes, TemplateDecodeLimits::default()).unwrap(),
            first
        );
    }

    #[test]
    fn rejects_unknown_references_and_schema_width_mismatch() {
        let mut bad_ref = source();
        bad_ref.entities[0].components[0].row = 99u32.to_le_bytes().to_vec();
        assert!(matches!(
            bake_scene(&bad_ref, BakeLimits::default()),
            Err(BakeError::UnknownReference { .. })
        ));

        let mut bad_width = source();
        bad_width.entities[0].components[0].row.pop();
        assert!(matches!(
            bake_scene(&bad_width, BakeLimits::default()),
            Err(BakeError::RowWidth { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_components_and_invalid_hierarchy() {
        let mut duplicate = source();
        let repeated = duplicate.entities[0].components[0].clone();
        duplicate.entities[0].components.push(repeated);
        assert!(matches!(
            bake_scene(&duplicate, BakeLimits::default()),
            Err(BakeError::DuplicateComponent { .. })
        ));

        let mut hierarchy = source();
        hierarchy.entities[1].parent = Some(LocalEntityId::new(1));
        assert!(matches!(
            bake_scene(&hierarchy, BakeLimits::default()),
            Err(BakeError::InvalidPlan(_))
        ));
    }

    #[test]
    fn encoded_output_obeys_runtime_decode_limits() {
        let bytes = bake_scene_bytes(&source(), BakeLimits::default()).unwrap();
        let limits = TemplateDecodeLimits {
            max_bytes: bytes.len() - 1,
            ..TemplateDecodeLimits::default()
        };
        assert!(decode_template(&bytes, limits).is_err());
    }

    #[test]
    fn source_metadata_obeys_bake_limits() {
        let source = source();
        let limits = BakeLimits {
            max_name_bytes: 1,
            ..BakeLimits::default()
        };
        assert!(matches!(
            bake_scene(&source, limits),
            Err(BakeError::LimitExceeded("name bytes"))
        ));
    }

    #[test]
    fn source_codec_round_trips_canonically_and_rejects_untrusted_counts() {
        let source = source();
        let encoded = encode_scene_source(&source, SceneSourceCodecLimits::default()).unwrap();
        let decoded = decode_scene_source(&encoded, SceneSourceCodecLimits::default()).unwrap();
        assert_eq!(
            bake_scene(&decoded, BakeLimits::default()).unwrap(),
            bake_scene(&source, BakeLimits::default()).unwrap()
        );

        let mut reordered = source.clone();
        reordered.entities.reverse();
        reordered.entities[0].components.reverse();
        let reordered_encoded =
            encode_scene_source(&reordered, SceneSourceCodecLimits::default()).unwrap();
        assert_eq!(encoded, reordered_encoded);

        let mut malformed = SOURCE_MAGIC.to_vec();
        malformed.extend_from_slice(&SOURCE_VERSION.to_le_bytes());
        malformed.extend_from_slice(&100u128.to_le_bytes());
        malformed.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode_scene_source(&malformed, SceneSourceCodecLimits::default()),
            Err(SceneSourceCodecError::LimitExceeded("schemas"))
                | Err(SceneSourceCodecError::Invalid("truncated count payload"))
        ));
    }
}
