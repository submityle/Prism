//! Stable, bounded prefab composition and reversible authoring transactions.
//!
//! Persistent IDs and exact semantic field paths replace runtime entity indices.
//! Arrays are atomic values. No engine, transport or UI dependency is required.

use crate::{AffectSet, Command, CommandError, EditContext, MergeResult};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::fmt;

macro_rules! identity {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(u128);
        impl $name {
            /// Reconstructs a caller-assigned persistent identity.
            #[must_use]
            pub const fn from_raw(raw: u128) -> Self {
                Self(raw)
            }
            /// Exact persisted identity.
            #[must_use]
            pub const fn raw(self) -> u128 {
                self.0
            }
        }
    };
}
identity!(PrefabId, "Stable asset identity, independent of file name.");
identity!(
    PrefabInstanceId,
    "Stable identity of a scene-level instance."
);
identity!(
    PrefabEntityId,
    "Stable source entity identity within one asset."
);
identity!(
    PrefabSlotId,
    "Stable nested-prefab slot identity within one asset."
);

/// Exact portable semantic value, independent of JSON or Rust memory layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefabValue {
    /// Null, distinct from a removed property.
    Null,
    /// Boolean.
    Bool(bool),
    /// Signed integer.
    Signed(i64),
    /// Unsigned integer.
    Unsigned(u64),
    /// Exact IEEE-754 f64 bits, including NaN payloads and negative zero.
    FloatBits(u64),
    /// Exact UTF-8 without trimming/normalization.
    String(String),
    /// Exact opaque bytes, also usable for wide scalar encodings.
    Bytes(Vec<u8>),
    /// Atomic sequence; overrides cannot navigate unstable array indices.
    Array(Vec<Self>),
    /// Semantic named fields/keys in lexical order.
    Object(BTreeMap<String, Self>),
}

/// Collision-free address: nested slot chain plus persistent source entity ID.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PrefabEntityPath {
    /// Slot chain from the containing asset to the source asset.
    pub slots: Vec<PrefabSlotId>,
    /// Source entity identity.
    pub entity: PrefabEntityId,
}
impl PrefabEntityPath {
    /// Addresses a locally authored entity.
    #[must_use]
    pub fn local(entity: PrefabEntityId) -> Self {
        Self {
            slots: Vec::new(),
            entity,
        }
    }
    fn prefixed(&self, prefix: &[PrefabSlotId]) -> Self {
        Self {
            slots: prefix.iter().chain(&self.slots).copied().collect(),
            entity: self.entity,
        }
    }
}

/// Stable component/field address. Empty fields address the entire component.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PrefabProperty {
    /// Qualified stable entity.
    pub entity: PrefabEntityPath,
    /// Stable host schema/component key.
    pub component: String,
    /// Exact object keys; dots, whitespace, slashes and Unicode are literal.
    pub fields: Vec<String>,
}
impl PrefabProperty {
    fn overlaps(&self, other: &Self) -> bool {
        self.entity == other.entity
            && self.component == other.component
            && (self.fields.starts_with(&other.fields) || other.fields.starts_with(&self.fields))
    }
}

/// Authored/resolved entity. Immutable component values are shared with history.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrefabEntity {
    /// Qualified parent; final composition rejects dangling/cyclic parents.
    pub parent: Option<PrefabEntityPath>,
    /// Stable component keys and exact semantic values.
    pub components: BTreeMap<String, Arc<PrefabValue>>,
}

/// Nested asset reference with optional parent in the containing namespace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefabSlot {
    /// Nested asset or variant.
    pub prefab: PrefabId,
    /// Parent assigned to every root entity of the nested composition.
    pub parent: Option<PrefabEntityPath>,
}

/// Explicit override; equality with the inherited value still represents a pin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefabOverride {
    /// Replace/add one component or named field.
    Set(Arc<PrefabValue>),
    /// Remove an existing component/field; missing targets are conflicts.
    Remove,
}

/// Authoring layer applied in order: slots, additions, removals, parents, properties.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrefabLayer {
    /// Local additions; inherited ID collisions are conflicts.
    pub entities: BTreeMap<PrefabEntityId, PrefabEntity>,
    /// Add/replace slots, or remove inherited slots with None.
    pub slots: BTreeMap<PrefabSlotId, Option<PrefabSlot>>,
    /// Explicit entity removals. Children must be removed/reparented explicitly.
    pub removed: BTreeSet<PrefabEntityPath>,
    /// Explicit parent replacements for surviving local or inherited entities.
    pub parents: BTreeMap<PrefabEntityPath, Option<PrefabEntityPath>>,
    /// Explicit non-overlapping semantic property overrides.
    pub overrides: BTreeMap<PrefabProperty, PrefabOverride>,
}

/// Base asset or single-parent variant. All inheritance/nesting edges form a DAG.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefabAsset {
    /// Persistent identity.
    pub id: PrefabId,
    /// Variant base, if any.
    pub base: Option<PrefabId>,
    /// Authored delta over that base.
    pub layer: PrefabLayer,
}

/// Scene instance and its explicit local override layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefabInstance {
    /// Persistent scene identity.
    pub id: PrefabInstanceId,
    /// Referenced base or variant.
    pub prefab: PrefabId,
    /// Local structural/property edits and nested references.
    pub layer: PrefabLayer,
}

/// Layer owner, suitable for inspector override queries.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PrefabOwner {
    /// Asset or variant.
    Asset(PrefabId),
    /// Scene instance.
    Instance(PrefabInstanceId),
}

/// Deterministically composed read model; namespace IDs remain lossless.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrefabComposition {
    /// Resolved entities.
    pub entities: BTreeMap<PrefabEntityPath, PrefabEntity>,
    /// Effective top-level slots, including inherited references.
    pub slots: BTreeMap<PrefabSlotId, PrefabSlot>,
}
impl PrefabComposition {
    /// Reads an exact property; missing and Null remain distinct.
    #[must_use]
    pub fn value(&self, property: &PrefabProperty) -> Option<&PrefabValue> {
        let mut value = self
            .entities
            .get(&property.entity)?
            .components
            .get(&property.component)?
            .as_ref();
        for key in &property.fields {
            let PrefabValue::Object(fields) = value else {
                return None;
            };
            value = fields.get(key)?;
        }
        Some(value)
    }
}

/// Limits for authoring, expansion, traversal, transactions and persisted files.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefabLimits {
    /// Maximum assets.
    pub max_assets: usize,
    /// Maximum scene instances.
    pub max_instances: usize,
    /// Aggregate expanded entity visits per validation/read.
    pub max_entities: usize,
    /// Aggregate semantic value nodes per operation.
    pub max_value_nodes: usize,
    /// Aggregate UTF-8/byte payload per operation.
    pub max_value_bytes: usize,
    /// Maximum dependency/field/value depth, no greater than 128.
    pub max_depth: usize,
    /// Maximum traversal/composition steps per operation.
    pub max_steps: usize,
    /// Maximum edits in one atomic transaction.
    pub max_edits: usize,
    /// Maximum encoded file size.
    pub max_codec_bytes: usize,
}
impl Default for PrefabLimits {
    fn default() -> Self {
        Self {
            max_assets: 1024,
            max_instances: 4096,
            max_entities: 65_536,
            max_value_nodes: 1_048_576,
            max_value_bytes: 32 * 1024 * 1024,
            max_depth: 64,
            max_steps: 2_000_000,
            max_edits: 4096,
            max_codec_bytes: 32 * 1024 * 1024,
        }
    }
}

/// Explicit failure; failed transactions preserve authoring, history and revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefabError {
    /// Missing asset, instance, entity, field or slot.
    Missing(String),
    /// Concurrent edit, overlapping override, duplicate identity or invalid base.
    Conflict(String),
    /// Dependency or parent cycle.
    Cycle,
    /// Invalid shape, path, limits or encoded input.
    Invalid(String),
    /// Named resource budget exhausted.
    Budget(&'static str),
    /// Optimistic revision mismatch.
    Revision {
        /// Required revision.
        expected: u64,
        /// Actual revision.
        actual: u64,
    },
}
impl fmt::Display for PrefabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(s) => write!(f, "missing prefab target: {s}"),
            Self::Conflict(s) => write!(f, "prefab conflict: {s}"),
            Self::Cycle => f.write_str("prefab dependency/parent cycle"),
            Self::Invalid(s) => write!(f, "invalid prefab data: {s}"),
            Self::Budget(s) => write!(f, "prefab budget exceeded: {s}"),
            Self::Revision { expected, actual } => write!(
                f,
                "prefab revision conflict: expected {expected}, actual {actual}"
            ),
        }
    }
}
impl core::error::Error for PrefabError {}
type Result<T> = core::result::Result<T, PrefabError>;
fn missing(s: impl Into<String>) -> PrefabError {
    PrefabError::Missing(s.into())
}
fn conflict(s: impl Into<String>) -> PrefabError {
    PrefabError::Conflict(s.into())
}
fn invalid(s: impl Into<String>) -> PrefabError {
    PrefabError::Invalid(s.into())
}

struct Meter {
    limits: PrefabLimits,
    steps: usize,
    entities: usize,
    nodes: usize,
    bytes: usize,
}
impl Meter {
    fn new(limits: PrefabLimits) -> Self {
        Self {
            limits,
            steps: 0,
            entities: 0,
            nodes: 0,
            bytes: 0,
        }
    }
    fn add(current: &mut usize, amount: usize, limit: usize, name: &'static str) -> Result<()> {
        *current = current
            .checked_add(amount)
            .ok_or(PrefabError::Budget(name))?;
        if *current > limit {
            return Err(PrefabError::Budget(name));
        }
        Ok(())
    }
    fn step(&mut self) -> Result<()> {
        Self::add(&mut self.steps, 1, self.limits.max_steps, "work")
    }
    fn depth(&self, depth: usize) -> Result<()> {
        if depth > self.limits.max_depth {
            Err(PrefabError::Budget("depth"))
        } else {
            Ok(())
        }
    }
    fn text(&mut self, text: &str) -> Result<()> {
        Self::add(
            &mut self.bytes,
            text.len(),
            self.limits.max_value_bytes,
            "value bytes",
        )
    }
    fn value(&mut self, value: &PrefabValue, depth: usize) -> Result<()> {
        self.depth(depth)?;
        self.step()?;
        Self::add(
            &mut self.nodes,
            1,
            self.limits.max_value_nodes,
            "value nodes",
        )?;
        match value {
            PrefabValue::String(s) => self.text(s)?,
            PrefabValue::Bytes(b) => Self::add(
                &mut self.bytes,
                b.len(),
                self.limits.max_value_bytes,
                "value bytes",
            )?,
            PrefabValue::Array(a) => {
                for v in a {
                    self.value(v, depth + 1)?;
                }
            }
            PrefabValue::Object(o) => {
                for (k, v) in o {
                    self.text(k)?;
                    self.value(v, depth + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn path(&mut self, p: &PrefabEntityPath) -> Result<()> {
        self.depth(p.slots.len())?;
        self.step()
    }
    fn property(&mut self, p: &PrefabProperty) -> Result<()> {
        self.path(&p.entity)?;
        self.depth(p.fields.len())?;
        if p.component.is_empty() {
            return Err(invalid("empty component key"));
        }
        self.text(&p.component)?;
        for f in &p.fields {
            self.step()?;
            self.text(f)?;
        }
        Ok(())
    }
    fn entity(&mut self, path: &PrefabEntityPath, e: &PrefabEntity) -> Result<()> {
        Self::add(
            &mut self.entities,
            1,
            self.limits.max_entities,
            "expanded entities",
        )?;
        self.path(path)?;
        if let Some(p) = &e.parent {
            self.path(p)?;
        }
        for (k, v) in &e.components {
            if k.is_empty() {
                return Err(invalid("empty component key"));
            }
            self.text(k)?;
            self.value(v, 0)?;
        }
        Ok(())
    }
}

/// Authoritative library. Immutable record sharing keeps inverse history sparse.
pub struct PrefabStore {
    assets: BTreeMap<PrefabId, Arc<PrefabAsset>>,
    instances: BTreeMap<PrefabInstanceId, Arc<PrefabInstance>>,
    revision: u64,
    limits: PrefabLimits,
    identity: Arc<()>,
}
impl EditContext for PrefabStore {}
impl PrefabStore {
    /// Creates an empty library, rejecting invalid limits.
    pub fn new(limits: PrefabLimits) -> Result<Self> {
        if [
            limits.max_assets,
            limits.max_instances,
            limits.max_entities,
            limits.max_value_nodes,
            limits.max_value_bytes,
            limits.max_depth,
            limits.max_steps,
            limits.max_edits,
            limits.max_codec_bytes,
        ]
        .contains(&0)
            || limits.max_depth > 128
        {
            return Err(invalid("limits must be nonzero; max_depth must be <= 128"));
        }
        Ok(Self {
            assets: BTreeMap::new(),
            instances: BTreeMap::new(),
            revision: 0,
            limits,
            identity: Arc::new(()),
        })
    }
    /// Monotonic revision advanced on commit, undo and redo.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Immutable asset; references remain valid through edits.
    #[must_use]
    pub fn asset(&self, id: PrefabId) -> Option<&Arc<PrefabAsset>> {
        self.assets.get(&id)
    }
    /// Immutable instance.
    #[must_use]
    pub fn instance(&self, id: PrefabInstanceId) -> Option<&Arc<PrefabInstance>> {
        self.instances.get(&id)
    }
    /// Authored assets in persistent identity order, including variants.
    pub fn assets(&self) -> impl Iterator<Item = &Arc<PrefabAsset>> {
        self.assets.values()
    }
    /// Authored scene instances in persistent identity order.
    pub fn instances(&self) -> impl Iterator<Item = &Arc<PrefabInstance>> {
        self.instances.values()
    }
    /// Admission/read/codec budgets configured for this library.
    #[must_use]
    pub const fn limits(&self) -> PrefabLimits {
        self.limits
    }
    /// Returns explicit overrides, rather than guessing from value inequality.
    pub fn layer(&self, owner: PrefabOwner) -> Result<&PrefabLayer> {
        match owner {
            PrefabOwner::Asset(id) => self.assets.get(&id).map(|a| &a.layer),
            PrefabOwner::Instance(id) => self.instances.get(&id).map(|i| &i.layer),
        }
        .ok_or_else(|| missing("layer owner"))
    }
    /// Resolves current dependencies; unrelated instance overrides stay pinned.
    pub fn compose(&self, owner: PrefabOwner) -> Result<PrefabComposition> {
        let overlay = Overlay::default();
        let view = View {
            store: self,
            overlay: &overlay,
        };
        Composer::new(view).owner(owner)
    }
}

#[derive(Default)]
struct Overlay {
    assets: BTreeMap<PrefabId, Option<Arc<PrefabAsset>>>,
    instances: BTreeMap<PrefabInstanceId, Option<Arc<PrefabInstance>>>,
}
#[derive(Clone, Copy)]
struct View<'a> {
    store: &'a PrefabStore,
    overlay: &'a Overlay,
}
impl<'a> View<'a> {
    fn asset(&self, id: PrefabId) -> Result<&'a Arc<PrefabAsset>> {
        self.overlay
            .assets
            .get(&id)
            .map_or_else(|| self.store.assets.get(&id), Option::as_ref)
            .ok_or_else(|| missing(format!("asset {}", id.raw())))
    }
    fn instance(&self, id: PrefabInstanceId) -> Result<&'a Arc<PrefabInstance>> {
        self.overlay
            .instances
            .get(&id)
            .map_or_else(|| self.store.instances.get(&id), Option::as_ref)
            .ok_or_else(|| missing(format!("instance {}", id.raw())))
    }
    fn layer(&self, owner: PrefabOwner) -> Result<&'a PrefabLayer> {
        Ok(match owner {
            PrefabOwner::Asset(id) => &self.asset(id)?.layer,
            PrefabOwner::Instance(id) => &self.instance(id)?.layer,
        })
    }
    fn prefab(&self, owner: PrefabOwner) -> Result<PrefabId> {
        match owner {
            PrefabOwner::Asset(id) => {
                self.asset(id)?;
                Ok(id)
            }
            PrefabOwner::Instance(id) => Ok(self.instance(id)?.prefab),
        }
    }
    fn descends(&self, mut id: PrefabId, ancestor: PrefabId, meter: &mut Meter) -> Result<bool> {
        let mut seen = BTreeSet::new();
        loop {
            meter.step()?;
            if !seen.insert(id) {
                return Err(PrefabError::Cycle);
            }
            meter.depth(seen.len())?;
            if id == ancestor {
                return Ok(true);
            }
            let Some(base) = self.asset(id)?.base else {
                return Ok(false);
            };
            id = base;
        }
    }
}

struct Composer<'a> {
    view: View<'a>,
    meter: Meter,
    active: Vec<PrefabId>,
}
impl<'a> Composer<'a> {
    fn new(view: View<'a>) -> Self {
        Self {
            meter: Meter::new(view.store.limits),
            view,
            active: Vec::new(),
        }
    }
    fn owner(&mut self, owner: PrefabOwner) -> Result<PrefabComposition> {
        match owner {
            PrefabOwner::Asset(id) => self.asset(id),
            PrefabOwner::Instance(id) => {
                let i = self.view.instance(id)?;
                let mut result = self.asset(i.prefab)?;
                self.apply(&mut result, &i.layer)?;
                Ok(result)
            }
        }
    }
    fn asset(&mut self, id: PrefabId) -> Result<PrefabComposition> {
        self.meter.step()?;
        self.meter.depth(self.active.len() + 1)?;
        if self.active.contains(&id) {
            return Err(PrefabError::Cycle);
        }
        let a = self.view.asset(id)?;
        self.active.push(id);
        let mut result = if let Some(base) = a.base {
            self.asset(base)?
        } else {
            PrefabComposition::default()
        };
        self.apply(&mut result, &a.layer)?;
        self.active.pop();
        Ok(result)
    }
    fn apply(&mut self, result: &mut PrefabComposition, layer: &PrefabLayer) -> Result<()> {
        validate_layer(layer, &mut self.meter)?;
        for (&id, slot) in &layer.slots {
            self.meter.step()?;
            if slot.is_none() && !result.slots.contains_key(&id) {
                return Err(missing("removed nested slot"));
            }
            Meter::add(
                &mut self.meter.steps,
                result.entities.len(),
                self.meter.limits.max_steps,
                "work",
            )?;
            result.entities.retain(|p, _| p.slots.first() != Some(&id));
            result.slots.remove(&id);
            if let Some(slot) = slot {
                let nested = self.asset(slot.prefab)?;
                for (path, mut entity) in nested.entities {
                    let path = path.prefixed(&[id]);
                    entity.parent = entity
                        .parent
                        .map(|p| p.prefixed(&[id]))
                        .or_else(|| slot.parent.clone());
                    self.meter.entity(&path, &entity)?;
                    if result.entities.insert(path, entity).is_some() {
                        return Err(conflict("nested entity collision"));
                    }
                }
                result.slots.insert(id, slot.clone());
            }
        }
        for (&id, entity) in &layer.entities {
            let path = PrefabEntityPath::local(id);
            self.meter.entity(&path, entity)?;
            if result.entities.insert(path, entity.clone()).is_some() {
                return Err(conflict("source entity ID already exists"));
            }
        }
        for path in &layer.removed {
            self.meter.step()?;
            if result.entities.remove(path).is_none() {
                return Err(missing("removed entity"));
            }
        }
        for (path, parent) in &layer.parents {
            self.meter.path(path)?;
            result
                .entities
                .get_mut(path)
                .ok_or_else(|| missing("reparent entity"))?
                .parent = parent.clone();
        }
        for (p, v) in &layer.overrides {
            apply_property(result, p, v, &mut self.meter)?;
        }
        for (path, e) in &result.entities {
            self.meter.entity(path, e)?;
            let mut cursor = e.parent.as_ref();
            let mut seen = BTreeSet::new();
            seen.insert(path);
            while let Some(parent) = cursor {
                self.meter.step()?;
                if !seen.insert(parent) {
                    return Err(PrefabError::Cycle);
                }
                self.meter.depth(seen.len())?;
                cursor = result
                    .entities
                    .get(parent)
                    .ok_or_else(|| missing("entity parent"))?
                    .parent
                    .as_ref();
            }
        }
        Ok(())
    }
}

fn validate_layer(layer: &PrefabLayer, meter: &mut Meter) -> Result<()> {
    for (id, e) in &layer.entities {
        meter.entity(&PrefabEntityPath::local(*id), e)?;
    }
    for slot in layer.slots.values() {
        meter.step()?;
        if let Some(PrefabSlot {
            parent: Some(p), ..
        }) = slot
        {
            meter.path(p)?;
        }
    }
    for path in &layer.removed {
        meter.path(path)?;
    }
    for (path, parent) in &layer.parents {
        meter.path(path)?;
        if let Some(parent) = parent {
            meter.path(parent)?;
        }
    }
    let mut previous: Option<&PrefabProperty> = None;
    for (p, v) in &layer.overrides {
        meter.property(p)?;
        if previous.is_some_and(|q| q.overlaps(p)) {
            return Err(conflict("overlapping overrides in one layer"));
        }
        if let PrefabOverride::Set(v) = v {
            meter.value(v, 0)?;
        }
        previous = Some(p);
    }
    Ok(())
}

fn apply_property(
    result: &mut PrefabComposition,
    p: &PrefabProperty,
    v: &PrefabOverride,
    meter: &mut Meter,
) -> Result<()> {
    meter.property(p)?;
    let entity = result
        .entities
        .get_mut(&p.entity)
        .ok_or_else(|| missing("override entity"))?;
    if p.fields.is_empty() {
        match v {
            PrefabOverride::Set(v) => {
                meter.value(v, 0)?;
                entity.components.insert(p.component.clone(), v.clone());
            }
            PrefabOverride::Remove => {
                entity
                    .components
                    .remove(&p.component)
                    .ok_or_else(|| missing("removed component"))?;
            }
        }
        return Ok(());
    }
    let old = entity
        .components
        .get(&p.component)
        .ok_or_else(|| missing("override component"))?;
    meter.value(old, 0)?;
    let mut root = old.as_ref().clone();
    let mut cursor = &mut root;
    for key in &p.fields[..p.fields.len() - 1] {
        let PrefabValue::Object(fields) = cursor else {
            return Err(conflict("field path crosses a non-object"));
        };
        cursor = fields
            .get_mut(key)
            .ok_or_else(|| missing("intermediate field"))?;
    }
    let PrefabValue::Object(fields) = cursor else {
        return Err(conflict("field parent is not an object"));
    };
    let key = p.fields.last().expect("nonempty path");
    match v {
        PrefabOverride::Set(v) => {
            meter.value(v, 0)?;
            fields.insert(key.clone(), v.as_ref().clone());
        }
        PrefabOverride::Remove => {
            fields.remove(key).ok_or_else(|| missing("removed field"))?;
        }
    }
    entity
        .components
        .insert(p.component.clone(), Arc::new(root));
    Ok(())
}

/// One authored edit. A slice is validated and committed atomically.
#[derive(Clone, Debug)]
pub enum PrefabEdit {
    /// Insert a new asset; duplicate IDs conflict.
    InsertAsset(Arc<PrefabAsset>),
    /// Replace an existing asset, including structural edits/reparenting.
    ReplaceAsset(Arc<PrefabAsset>),
    /// Remove an asset; surviving references reject the whole batch.
    RemoveAsset(PrefabId),
    /// Insert a scene instance.
    InsertInstance(Arc<PrefabInstance>),
    /// Replace a scene instance's authored record.
    ReplaceInstance(Arc<PrefabInstance>),
    /// Remove a scene instance.
    RemoveInstance(PrefabInstanceId),
    /// Author an explicit value pin or removal.
    Override {
        /// Layer to edit.
        owner: PrefabOwner,
        /// Exact target.
        property: PrefabProperty,
        /// Explicit value/removal.
        value: PrefabOverride,
    },
    /// Remove an exact authored override, exposing inheritance.
    Revert {
        /// Layer to edit.
        owner: PrefabOwner,
        /// Exact authored target.
        property: PrefabProperty,
    },
    /// Move an exact override to an ancestor asset; intervening overlaps conflict.
    ApplyToBase {
        /// Source layer.
        source: PrefabOwner,
        /// Explicit ancestor asset (same qualified namespace).
        target: PrefabId,
        /// Exact authored override.
        property: PrefabProperty,
    },
    /// Pin the source's effective value/absence into selected descendant layers.
    /// Selecting a variant intentionally affects its inheriting descendants.
    Propagate {
        /// Source asset.
        source: PrefabId,
        /// Qualified property in the source namespace.
        property: PrefabProperty,
        /// Unique proper descendants, bounded as edits.
        recipients: Vec<PrefabOwner>,
    },
}

type AssetChange = (Option<Arc<PrefabAsset>>, Option<Arc<PrefabAsset>>);
type InstanceChange = (Option<Arc<PrefabInstance>>, Option<Arc<PrefabInstance>>);

/// Sparse inverse/forward history retaining only changed authored records.
pub struct PrefabReceipt {
    identity: Arc<()>,
    assets: BTreeMap<PrefabId, AssetChange>,
    instances: BTreeMap<PrefabInstanceId, InstanceChange>,
}
impl PrefabReceipt {
    /// Changed authoring owners; dependent instances may also compose differently.
    pub fn owners(&self) -> impl Iterator<Item = PrefabOwner> + '_ {
        self.assets
            .keys()
            .copied()
            .map(PrefabOwner::Asset)
            .chain(self.instances.keys().copied().map(PrefabOwner::Instance))
    }
    /// Number of changed records; independent of unaffected library size.
    #[must_use]
    pub fn changed_records(&self) -> usize {
        self.assets.len() + self.instances.len()
    }
    /// Undo with exact record identity guards; unrelated edits may survive.
    pub fn undo(&self, store: &mut PrefabStore) -> Result<()> {
        self.restore(store, false)
    }
    /// Redo with exact record identity guards.
    pub fn redo(&self, store: &mut PrefabStore) -> Result<()> {
        self.restore(store, true)
    }
    fn restore(&self, store: &mut PrefabStore, forward: bool) -> Result<()> {
        if !Arc::ptr_eq(&self.identity, &store.identity) {
            return Err(conflict("history belongs to another store"));
        }
        let mut overlay = Overlay::default();
        for (&id, (before, after)) in &self.assets {
            let (expected, replacement) = if forward {
                (before, after)
            } else {
                (after, before)
            };
            if !same_record(store.assets.get(&id), expected.as_ref()) {
                return Err(conflict("asset changed since history entry"));
            }
            overlay.assets.insert(id, replacement.clone());
        }
        for (&id, (before, after)) in &self.instances {
            let (expected, replacement) = if forward {
                (before, after)
            } else {
                (after, before)
            };
            if !same_record(store.instances.get(&id), expected.as_ref()) {
                return Err(conflict("instance changed since history entry"));
            }
            overlay.instances.insert(id, replacement.clone());
        }
        validate_view(
            View {
                store,
                overlay: &overlay,
            },
            Meter::new(store.limits),
        )?;
        store.commit_overlay(overlay)?;
        Ok(())
    }
}
fn same_record<T>(a: Option<&Arc<T>>, b: Option<&Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

impl PrefabStore {
    /// Apply a bounded atomic batch with optimistic revision fencing.
    /// No record or revision changes on any validation/conflict/budget failure.
    pub fn transact(&mut self, expected: u64, edits: &[PrefabEdit]) -> Result<PrefabReceipt> {
        if expected != self.revision {
            return Err(PrefabError::Revision {
                expected,
                actual: self.revision,
            });
        }
        if edits.len() > self.limits.max_edits {
            return Err(PrefabError::Budget("edits"));
        }
        let mut overlay = Overlay::default();
        let mut meter = Meter::new(self.limits);
        let mut count = edits.len();
        for edit in edits {
            stage(self, &mut overlay, edit, &mut meter, &mut count)?;
        }
        validate_view(
            View {
                store: self,
                overlay: &overlay,
            },
            meter,
        )?;
        // Preserve original record identities for no-ops and previous undo entries.
        overlay
            .assets
            .retain(|id, a| self.assets.get(id).map(Arc::as_ref) != a.as_deref());
        overlay
            .instances
            .retain(|id, i| self.instances.get(id).map(Arc::as_ref) != i.as_deref());
        let receipt = PrefabReceipt {
            identity: self.identity.clone(),
            assets: overlay
                .assets
                .iter()
                .map(|(&id, a)| (id, (self.assets.get(&id).cloned(), a.clone())))
                .collect(),
            instances: overlay
                .instances
                .iter()
                .map(|(&id, i)| (id, (self.instances.get(&id).cloned(), i.clone())))
                .collect(),
        };
        self.commit_overlay(overlay)?;
        Ok(receipt)
    }
    fn commit_overlay(&mut self, overlay: Overlay) -> Result<()> {
        if overlay.assets.is_empty() && overlay.instances.is_empty() {
            return Ok(());
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(PrefabError::Budget("revision"))?;
        for (id, a) in overlay.assets {
            if let Some(a) = a {
                self.assets.insert(id, a);
            } else {
                self.assets.remove(&id);
            }
        }
        for (id, i) in overlay.instances {
            if let Some(i) = i {
                self.instances.insert(id, i);
            } else {
                self.instances.remove(&id);
            }
        }
        self.revision = revision;
        Ok(())
    }
}

fn validate_view(view: View<'_>, meter: Meter) -> Result<()> {
    // Union allocation is bounded by already admitted store + bounded edit batch.
    let assets: BTreeSet<_> = view
        .store
        .assets
        .keys()
        .chain(view.overlay.assets.keys())
        .copied()
        .collect();
    let instances: BTreeSet<_> = view
        .store
        .instances
        .keys()
        .chain(view.overlay.instances.keys())
        .copied()
        .collect();
    let asset_count = assets.iter().filter(|&&id| view.asset(id).is_ok()).count();
    let instance_count = instances
        .iter()
        .filter(|&&id| view.instance(id).is_ok())
        .count();
    if asset_count > view.store.limits.max_assets {
        return Err(PrefabError::Budget("assets"));
    }
    if instance_count > view.store.limits.max_instances {
        return Err(PrefabError::Budget("instances"));
    }
    let mut composer = Composer {
        view,
        meter,
        active: Vec::new(),
    };
    for id in assets {
        if view.asset(id).is_ok() {
            composer.owner(PrefabOwner::Asset(id))?;
        }
    }
    for id in instances {
        if view.instance(id).is_ok() {
            composer.owner(PrefabOwner::Instance(id))?;
        }
    }
    Ok(())
}

fn edit_layer(
    store: &PrefabStore,
    overlay: &mut Overlay,
    owner: PrefabOwner,
    meter: &mut Meter,
    edit: impl FnOnce(&mut PrefabLayer) -> Result<()>,
) -> Result<()> {
    let view = View { store, overlay };
    validate_layer(view.layer(owner)?, meter)?; // Charge before cloning an authored shell.
    match owner {
        PrefabOwner::Asset(id) => {
            let mut record = view.asset(id)?.as_ref().clone();
            edit(&mut record.layer)?;
            overlay.assets.insert(id, Some(Arc::new(record)));
        }
        PrefabOwner::Instance(id) => {
            let mut record = view.instance(id)?.as_ref().clone();
            edit(&mut record.layer)?;
            overlay.instances.insert(id, Some(Arc::new(record)));
        }
    }
    Ok(())
}

fn stage(
    store: &PrefabStore,
    overlay: &mut Overlay,
    edit: &PrefabEdit,
    meter: &mut Meter,
    count: &mut usize,
) -> Result<()> {
    meter.step()?;
    let view = View { store, overlay };
    match edit {
        PrefabEdit::InsertAsset(a) | PrefabEdit::ReplaceAsset(a) => {
            let present = view.asset(a.id).is_ok();
            if present != matches!(edit, PrefabEdit::ReplaceAsset(_)) {
                return Err(conflict("asset insert/replace existence"));
            }
            validate_layer(&a.layer, meter)?;
            overlay.assets.insert(a.id, Some(a.clone()));
        }
        PrefabEdit::RemoveAsset(id) => {
            view.asset(*id)?;
            overlay.assets.insert(*id, None);
        }
        PrefabEdit::InsertInstance(i) | PrefabEdit::ReplaceInstance(i) => {
            let present = view.instance(i.id).is_ok();
            if present != matches!(edit, PrefabEdit::ReplaceInstance(_)) {
                return Err(conflict("instance insert/replace existence"));
            }
            validate_layer(&i.layer, meter)?;
            overlay.instances.insert(i.id, Some(i.clone()));
        }
        PrefabEdit::RemoveInstance(id) => {
            view.instance(*id)?;
            overlay.instances.insert(*id, None);
        }
        PrefabEdit::Override {
            owner,
            property,
            value,
        } => {
            meter.property(property)?;
            if let PrefabOverride::Set(v) = value {
                meter.value(v, 0)?;
            }
            edit_layer(store, overlay, *owner, meter, |layer| {
                layer.overrides.insert(property.clone(), value.clone());
                Ok(())
            })?;
        }
        PrefabEdit::Revert { owner, property } => {
            meter.property(property)?;
            edit_layer(store, overlay, *owner, meter, |layer| {
                layer
                    .overrides
                    .remove(property)
                    .ok_or_else(|| missing("authored override"))?;
                Ok(())
            })?;
        }
        PrefabEdit::ApplyToBase {
            source,
            target,
            property,
        } => {
            meter.property(property)?;
            let source_id = view.prefab(*source)?;
            if *source == PrefabOwner::Asset(*target)
                || !view.descends(source_id, *target, meter)?
            {
                return Err(conflict("apply target is not a proper ancestor"));
            }
            let value = view
                .layer(*source)?
                .overrides
                .get(property)
                .ok_or_else(|| missing("authored override"))?
                .clone();
            // Skipping an intervening pin would make 'apply' misleading.
            let mut cursor = match source {
                PrefabOwner::Asset(_) => view.asset(source_id)?.base,
                PrefabOwner::Instance(_) => Some(source_id),
            };
            while let Some(id) = cursor {
                meter.step()?;
                if id == *target {
                    break;
                }
                let a = view.asset(id)?;
                Meter::add(
                    &mut meter.steps,
                    a.layer.overrides.len(),
                    meter.limits.max_steps,
                    "work",
                )?;
                if a.layer.overrides.keys().any(|p| p.overlaps(property)) {
                    return Err(conflict("intervening variant override"));
                }
                cursor = a.base;
            }
            edit_layer(store, overlay, *source, meter, |layer| {
                layer.overrides.remove(property);
                Ok(())
            })?;
            edit_layer(
                store,
                overlay,
                PrefabOwner::Asset(*target),
                meter,
                |layer| {
                    layer.overrides.insert(property.clone(), value);
                    Ok(())
                },
            )?;
        }
        PrefabEdit::Propagate {
            source,
            property,
            recipients,
        } => {
            Meter::add(count, recipients.len(), store.limits.max_edits, "edits")?;
            meter.property(property)?;
            let mut composer = Composer {
                view,
                meter: Meter::new(store.limits),
                active: Vec::new(),
            };
            let composition = composer.asset(*source)?;
            let value = composition.value(property).cloned().map(Arc::new);
            // Charge this traversal to the same batch budget.
            meter.absorb(&composer.meter)?;
            if !composition.entities.contains_key(&property.entity) {
                return Err(missing("propagation entity"));
            }
            let mut seen = BTreeSet::new();
            // Resolve/validate all recipients before modifying any layer.
            let mut ordered = Vec::new();
            for recipient in recipients {
                meter.step()?;
                if !seen.insert(*recipient) {
                    return Err(conflict("duplicate propagation recipient"));
                }
                if *recipient == PrefabOwner::Asset(*source)
                    || !view.descends(view.prefab(*recipient)?, *source, meter)?
                {
                    return Err(conflict("propagation recipient is unrelated"));
                }
                Meter::add(
                    &mut meter.steps,
                    view.layer(*recipient)?.overrides.len(),
                    meter.limits.max_steps,
                    "work",
                )?;
                if view
                    .layer(*recipient)?
                    .overrides
                    .keys()
                    .any(|p| p.overlaps(property) && p.fields.len() < property.fields.len())
                {
                    return Err(conflict(
                        "propagation would discard sibling fields of a broader override",
                    ));
                }
                // Ancestors before descendants makes absence propagation
                // independent of caller order (no redundant Remove tombstone).
                let mut cursor = view.prefab(*recipient)?;
                let mut distance = 0;
                while cursor != *source {
                    meter.step()?;
                    cursor = view
                        .asset(cursor)?
                        .base
                        .ok_or_else(|| conflict("propagation ancestry changed"))?;
                    distance += 1;
                    meter.depth(distance)?;
                }
                ordered.push((distance, *recipient));
            }
            ordered.sort();
            for (_, recipient) in &ordered {
                edit_layer(store, overlay, *recipient, meter, |layer| {
                    layer.overrides.retain(|p, _| !p.overlaps(property));
                    Ok(())
                })?;
                let mut composer = Composer::new(View { store, overlay });
                let inherited = composer.owner(*recipient)?;
                meter.absorb(&composer.meter)?;
                let replacement = match &value {
                    Some(v) => Some(PrefabOverride::Set(v.clone())),
                    None if inherited.value(property).is_some() => Some(PrefabOverride::Remove),
                    None => None,
                };
                if let Some(value) = replacement {
                    edit_layer(store, overlay, *recipient, meter, |layer| {
                        layer.overrides.insert(property.clone(), value);
                        Ok(())
                    })?;
                }
            }
        }
    }
    Ok(())
}
impl Meter {
    fn absorb(&mut self, other: &Self) -> Result<()> {
        Self::add(&mut self.steps, other.steps, self.limits.max_steps, "work")?;
        Self::add(
            &mut self.entities,
            other.entities,
            self.limits.max_entities,
            "expanded entities",
        )?;
        Self::add(
            &mut self.nodes,
            other.nodes,
            self.limits.max_value_nodes,
            "value nodes",
        )?;
        Self::add(
            &mut self.bytes,
            other.bytes,
            self.limits.max_value_bytes,
            "value bytes",
        )
    }
}

/// One atomic Prefab transaction integrated with the kernel's inverse history.
/// Retains changed records only; never snapshots/clones the whole store.
pub struct PrefabCommand {
    label: String,
    affected: AffectSet,
    expected: u64,
    edits: Vec<PrefabEdit>,
    receipt: Option<PrefabReceipt>,
    applied: bool,
}
impl PrefabCommand {
    /// Captures the caller's revision and bounded authoring request.
    #[must_use]
    pub fn new(label: impl Into<String>, expected: u64, edits: Vec<PrefabEdit>) -> Self {
        let mut affected = AffectSet::new();
        affected.insert("prefab");
        Self {
            label: label.into(),
            affected,
            expected,
            edits,
            receipt: None,
            applied: false,
        }
    }
}
impl Command<PrefabStore> for PrefabCommand {
    fn apply(&mut self, context: &mut PrefabStore) -> core::result::Result<(), CommandError> {
        if self.applied {
            return Err(CommandError::new("prefab command already applied"));
        }
        if let Some(receipt) = &self.receipt {
            receipt
                .redo(context)
                .map_err(|e| CommandError::new(e.to_string()))?;
        } else {
            self.receipt = Some(
                context
                    .transact(self.expected, &self.edits)
                    .map_err(|e| CommandError::new(e.to_string()))?,
            );
            self.edits.clear();
            self.edits.shrink_to_fit();
        }
        self.applied = true;
        Ok(())
    }
    fn undo(&mut self, context: &mut PrefabStore) -> core::result::Result<(), CommandError> {
        if !self.applied {
            return Err(CommandError::new("prefab command is not applied"));
        }
        self.receipt
            .as_ref()
            .ok_or_else(|| CommandError::new("missing prefab history"))?
            .undo(context)
            .map_err(|e| CommandError::new(e.to_string()))?;
        self.applied = false;
        Ok(())
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn affected(&self) -> &AffectSet {
        &self.affected
    }
    fn merge_from(&mut self, _next: &dyn Command<PrefabStore>) -> MergeResult {
        MergeResult::NotMergeable
    }
}

impl PrefabStore {
    /// Deterministic versioned binary encoding of authored records and revision.
    /// Does not encode expanded entities, history, limits or host runtime handles.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let overlay = Overlay::default();
        validate_view(
            View {
                store: self,
                overlay: &overlay,
            },
            Meter::new(self.limits),
        )?;
        let mut writer = Writer {
            bytes: Vec::new(),
            limit: self.limits.max_codec_bytes,
        };
        writer.put(b"PRPF\x01\0\0\0")?;
        self.revision.write(&mut writer)?;
        self.assets.write(&mut writer)?;
        self.instances.write(&mut writer)?;
        Ok(writer.bytes)
    }
    /// Decode into a new store, rejecting noncanonical, truncated, trailing,
    /// over-budget or invalid dependency data before publishing any state.
    pub fn decode(bytes: &[u8], limits: PrefabLimits) -> Result<Self> {
        let mut store = Self::new(limits)?;
        if bytes.len() > limits.max_codec_bytes {
            return Err(PrefabError::Budget("codec bytes"));
        }
        let mut reader = Reader {
            bytes,
            cursor: 0,
            meter: Meter::new(limits),
            value_depth: 0,
        };
        if reader.take(8)? != b"PRPF\x01\0\0\0" {
            return Err(invalid("codec magic/version"));
        }
        store.revision = u64::read(&mut reader)?;
        // Asset/instance counts are checked before allocating their maps.
        let count = reader.count()?;
        if count > limits.max_assets {
            return Err(PrefabError::Budget("assets"));
        }
        for _ in 0..count {
            let id = PrefabId::read(&mut reader)?;
            let a = Arc::<PrefabAsset>::read(&mut reader)?;
            if a.id != id {
                return Err(invalid("asset identity mismatch"));
            }
            ordered_insert(&mut store.assets, id, a)?;
        }
        let count = reader.count()?;
        if count > limits.max_instances {
            return Err(PrefabError::Budget("instances"));
        }
        for _ in 0..count {
            let id = PrefabInstanceId::read(&mut reader)?;
            let i = Arc::<PrefabInstance>::read(&mut reader)?;
            if i.id != id {
                return Err(invalid("instance identity mismatch"));
            }
            ordered_insert(&mut store.instances, id, i)?;
        }
        if reader.cursor != bytes.len() {
            return Err(invalid("trailing codec bytes"));
        }
        let overlay = Overlay::default();
        validate_view(
            View {
                store: &store,
                overlay: &overlay,
            },
            reader.meter,
        )?;
        Ok(store)
    }
}

struct Writer {
    bytes: Vec<u8>,
    limit: usize,
}
impl Writer {
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(PrefabError::Budget("codec bytes"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn count(&mut self, count: usize) -> Result<()> {
        u64::try_from(count)
            .map_err(|_| PrefabError::Budget("codec count"))?
            .write(self)
    }
    fn blob(&mut self, bytes: &[u8]) -> Result<()> {
        self.count(bytes.len())?;
        self.put(bytes)
    }
}
struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
    meter: Meter,
    value_depth: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        self.meter.step()?;
        let end = self
            .cursor
            .checked_add(count)
            .ok_or_else(|| invalid("codec length overflow"))?;
        let result = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| invalid("truncated codec"))?;
        self.cursor = end;
        Ok(result)
    }
    fn count(&mut self) -> Result<usize> {
        let count = usize::try_from(u64::read(self)?)
            .map_err(|_| invalid("unrepresentable codec count"))?;
        // Every sequence element consumes at least one byte. Never reserve an
        // attacker-supplied capacity or loop without a work/remaining-byte bound.
        if count > self.bytes.len().saturating_sub(self.cursor) {
            return Err(invalid("codec count exceeds input"));
        }
        if count > self.meter.limits.max_steps.saturating_sub(self.meter.steps) {
            return Err(PrefabError::Budget("work"));
        }
        Ok(count)
    }
    fn blob(&mut self) -> Result<&'a [u8]> {
        let len = usize::try_from(u64::read(self)?)
            .map_err(|_| invalid("unrepresentable blob length"))?;
        Meter::add(
            &mut self.meter.bytes,
            len,
            self.meter.limits.max_value_bytes,
            "value bytes",
        )?;
        self.take(len)
    }
}
trait Binary: Sized {
    fn write(&self, writer: &mut Writer) -> Result<()>;
    fn read(reader: &mut Reader<'_>) -> Result<Self>;
}
macro_rules! scalar {
    ($ty:ty,$size:literal) => {
        impl Binary for $ty {
            fn write(&self, w: &mut Writer) -> Result<()> {
                w.put(&self.to_le_bytes())
            }
            fn read(r: &mut Reader<'_>) -> Result<Self> {
                Ok(Self::from_le_bytes(
                    r.take($size)?
                        .try_into()
                        .map_err(|_| invalid("scalar size"))?,
                ))
            }
        }
    };
}
scalar!(u8, 1);
scalar!(u64, 8);
scalar!(i64, 8);
scalar!(u128, 16);
macro_rules! binary_id { ($($ty:ty),+) => { $(impl Binary for $ty {
    fn write(&self,w: &mut Writer) -> Result<()> { self.raw().write(w) }
    fn read(r: &mut Reader<'_>) -> Result<Self> { Ok(Self::from_raw(u128::read(r)?)) }
})+ }; }
binary_id!(PrefabId, PrefabInstanceId, PrefabEntityId, PrefabSlotId);
impl Binary for bool {
    fn write(&self, w: &mut Writer) -> Result<()> {
        u8::from(*self).write(w)
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        match u8::read(r)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("boolean tag")),
        }
    }
}
impl Binary for String {
    fn write(&self, w: &mut Writer) -> Result<()> {
        w.blob(self.as_bytes())
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        core::str::from_utf8(r.blob()?)
            .map(String::from)
            .map_err(|_| invalid("non UTF-8 string"))
    }
}
impl<T: Binary> Binary for Arc<T> {
    fn write(&self, w: &mut Writer) -> Result<()> {
        self.as_ref().write(w)
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        Ok(Arc::new(T::read(r)?))
    }
}
impl<T: Binary> Binary for Option<T> {
    fn write(&self, w: &mut Writer) -> Result<()> {
        self.is_some().write(w)?;
        if let Some(v) = self {
            v.write(w)?;
        }
        Ok(())
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        if bool::read(r)? {
            Ok(Some(T::read(r)?))
        } else {
            Ok(None)
        }
    }
}
impl<T: Binary> Binary for Vec<T> {
    fn write(&self, w: &mut Writer) -> Result<()> {
        w.count(self.len())?;
        for v in self {
            v.write(w)?;
        }
        Ok(())
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        let count = r.count()?;
        let mut result = Vec::new();
        for _ in 0..count {
            r.meter.step()?;
            result.push(T::read(r)?);
        }
        Ok(result)
    }
}
fn ordered_insert<K: Ord, V>(map: &mut BTreeMap<K, V>, key: K, value: V) -> Result<()> {
    if map.last_key_value().is_some_and(|(k, _)| k >= &key) {
        return Err(invalid("duplicate or unordered codec key"));
    }
    map.insert(key, value);
    Ok(())
}
impl<K: Binary + Ord, V: Binary> Binary for BTreeMap<K, V> {
    fn write(&self, w: &mut Writer) -> Result<()> {
        w.count(self.len())?;
        for (k, v) in self {
            k.write(w)?;
            v.write(w)?;
        }
        Ok(())
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        let count = r.count()?;
        let mut result = Self::new();
        for _ in 0..count {
            r.meter.step()?;
            let k = K::read(r)?;
            let v = V::read(r)?;
            ordered_insert(&mut result, k, v)?;
        }
        Ok(result)
    }
}
impl<T: Binary + Ord> Binary for BTreeSet<T> {
    fn write(&self, w: &mut Writer) -> Result<()> {
        w.count(self.len())?;
        for v in self {
            v.write(w)?;
        }
        Ok(())
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        let count = r.count()?;
        let mut result = Self::new();
        for _ in 0..count {
            r.meter.step()?;
            let value = T::read(r)?;
            if result.last().is_some_and(|v| v >= &value) {
                return Err(invalid("duplicate or unordered codec set"));
            }
            result.insert(value);
        }
        Ok(result)
    }
}
macro_rules! record {
    ($ty:ty { $($field:ident),+ }) => { impl Binary for $ty {
        fn write(&self,w: &mut Writer) -> Result<()> { $(self.$field.write(w)?;)+ Ok(()) }
        fn read(r: &mut Reader<'_>) -> Result<Self> { Ok(Self { $($field: Binary::read(r)?),+ }) }
    } };
}
record!(PrefabEntityPath { slots, entity });
record!(PrefabProperty {
    entity,
    component,
    fields
});
record!(PrefabEntity { parent, components });
record!(PrefabSlot { prefab, parent });
record!(PrefabLayer {
    entities,
    slots,
    removed,
    parents,
    overrides
});
record!(PrefabAsset { id, base, layer });
record!(PrefabInstance { id, prefab, layer });
impl Binary for PrefabOverride {
    fn write(&self, w: &mut Writer) -> Result<()> {
        match self {
            Self::Remove => 0u8.write(w),
            Self::Set(v) => {
                1u8.write(w)?;
                v.write(w)
            }
        }
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        match u8::read(r)? {
            0 => Ok(Self::Remove),
            1 => Ok(Self::Set(Binary::read(r)?)),
            _ => Err(invalid("override tag")),
        }
    }
}
impl Binary for PrefabValue {
    fn write(&self, w: &mut Writer) -> Result<()> {
        match self {
            Self::Null => 0u8.write(w),
            Self::Bool(v) => {
                1u8.write(w)?;
                v.write(w)
            }
            Self::Signed(v) => {
                2u8.write(w)?;
                v.write(w)
            }
            Self::Unsigned(v) => {
                3u8.write(w)?;
                v.write(w)
            }
            Self::FloatBits(v) => {
                4u8.write(w)?;
                v.write(w)
            }
            Self::String(v) => {
                5u8.write(w)?;
                v.write(w)
            }
            Self::Bytes(v) => {
                6u8.write(w)?;
                w.blob(v)
            }
            Self::Array(v) => {
                7u8.write(w)?;
                v.write(w)
            }
            Self::Object(v) => {
                8u8.write(w)?;
                v.write(w)
            }
        }
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        r.meter.depth(r.value_depth)?;
        Meter::add(
            &mut r.meter.nodes,
            1,
            r.meter.limits.max_value_nodes,
            "value nodes",
        )?;
        r.value_depth += 1;
        let value = match u8::read(r)? {
            0 => Self::Null,
            1 => Self::Bool(Binary::read(r)?),
            2 => Self::Signed(Binary::read(r)?),
            3 => Self::Unsigned(Binary::read(r)?),
            4 => Self::FloatBits(Binary::read(r)?),
            5 => Self::String(Binary::read(r)?),
            6 => Self::Bytes(r.blob()?.to_vec()),
            7 => Self::Array(Binary::read(r)?),
            8 => Self::Object(Binary::read(r)?),
            _ => return Err(invalid("value tag")),
        };
        r.value_depth -= 1;
        Ok(value)
    }
}
