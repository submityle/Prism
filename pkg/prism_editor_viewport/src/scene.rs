//! Renderer-neutral scene ID-pass identities and incremental submissions.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::fmt;

use crate::EncodedPickId;

/// Stable identity of an editor entity.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct EntityId(u128);

impl EntityId {
    /// Creates an entity identity from persisted data.
    #[must_use]
    pub const fn from_raw(value: u128) -> Self {
        Self(value)
    }

    /// Returns the persisted value.
    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Stable identity of a mesh resource.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct MeshId(u128);

impl MeshId {
    /// Creates a mesh identity from persisted data.
    #[must_use]
    pub const fn from_raw(value: u128) -> Self {
        Self(value)
    }

    /// Returns the persisted value.
    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Stable identity of a mesh subobject, such as a primitive or material slot.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct SubobjectId(u64);

impl SubobjectId {
    /// Creates a subobject identity from persisted data.
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Returns the persisted value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// The stable identity written by one ID-pass draw.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct SceneObjectId {
    /// Entity containing the rendered object.
    pub entity: EntityId,
    /// Mesh resource used by the draw.
    pub mesh: MeshId,
    /// Mesh subobject represented by the draw.
    pub subobject: SubobjectId,
}

impl SceneObjectId {
    /// Creates a scene object identity.
    #[must_use]
    pub const fn new(entity: EntityId, mesh: MeshId, subobject: SubobjectId) -> Self {
        Self {
            entity,
            mesh,
            subobject,
        }
    }
}

/// Renderer-owned stable handles needed to materialize one ID-pass draw.
///
/// The viewport crate does not interpret these handles. A renderer adapter maps
/// them to geometry, transforms, visibility, and pipeline state in its own data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SceneIdPassItem {
    /// Stable entity/mesh/subobject identity written to the ID target.
    pub id: SceneObjectId,
    /// Renderer-owned geometry or draw handle.
    pub draw_key: u64,
    /// Renderer-owned instance or transform handle.
    pub instance_key: u64,
    /// Whether this item participates in the current ID pass.
    pub visible: bool,
    /// Deterministic renderer sort key.
    pub sort_key: u64,
}

impl SceneIdPassItem {
    /// Creates one renderer-neutral ID-pass item.
    #[must_use]
    pub const fn new(
        id: SceneObjectId,
        draw_key: u64,
        instance_key: u64,
        visible: bool,
        sort_key: u64,
    ) -> Self {
        Self {
            id,
            draw_key,
            instance_key,
            visible,
            sort_key,
        }
    }
}

/// A scene item with its stable encoded GPU value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SceneIdPassEntry {
    item: SceneIdPassItem,
    encoded: EncodedPickId,
}

/// Renderer adapter invoked for deterministic incremental ID-pass updates.
///
/// Each added or updated entry contains both renderer-owned draw handles and
/// the exact integer value its fragment shader must write. The adapter owns
/// pipeline creation, geometry binding, and command encoding. The optional
/// [`crate::SceneIdGeometryRenderer`] is a concrete indexed-geometry adapter.
pub trait SceneIdPassSink {
    /// Adapter-specific failure produced while recording an addition.
    type Error;

    /// Records a new ID-pass draw.
    fn add(&mut self, entry: SceneIdPassEntry) -> Result<(), Self::Error>;

    /// Records an update to a previously known ID-pass draw.
    fn update(&mut self, entry: SceneIdPassEntry) -> Result<(), Self::Error>;

    /// Removes a prior ID-pass draw by its stable scene identity.
    fn remove(&mut self, id: SceneObjectId) -> Result<(), Self::Error>;
}

impl SceneIdPassEntry {
    /// Returns the renderer-neutral item.
    #[must_use]
    pub const fn item(self) -> SceneIdPassItem {
        self.item
    }

    /// Returns the value to write from the ID fragment shader.
    #[must_use]
    pub const fn encoded_id(self) -> EncodedPickId {
        self.encoded
    }
}

/// Failure while allocating or applying scene IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneIdPassError {
    /// The same scene object was supplied more than once in one snapshot.
    DuplicateObject(SceneObjectId),
    /// The same scene object was supplied more than once in one patch.
    DuplicatePatchKey(SceneObjectId),
    /// The monotonic token space is exhausted.
    TokenExhausted,
    /// The scene generation cannot advance further.
    GenerationExhausted,
}

impl fmt::Display for SceneIdPassError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateObject(id) => write!(
                formatter,
                "duplicate ID-pass object ({}/{}/{})",
                id.entity.raw(),
                id.mesh.raw(),
                id.subobject.raw()
            ),
            Self::DuplicatePatchKey(id) => write!(
                formatter,
                "duplicate ID-pass patch key ({}/{}/{})",
                id.entity.raw(),
                id.mesh.raw(),
                id.subobject.raw()
            ),
            Self::TokenExhausted => formatter.write_str("ID-pass token space exhausted"),
            Self::GenerationExhausted => formatter.write_str("ID-pass generation exhausted"),
        }
    }
}

impl core::error::Error for SceneIdPassError {}

/// One incremental scene mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneIdPassOperation {
    /// Adds a new object or replaces the renderer handles of an existing one.
    Upsert(SceneIdPassItem),
    /// Removes an object from the current ID-pass scene.
    Remove(SceneObjectId),
}

/// Deterministic changes produced by one scene submission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneIdPassDelta {
    /// New objects that must be added to the renderer's ID-pass draw list.
    pub added: Vec<SceneIdPassEntry>,
    /// Existing objects whose renderer handles or visibility changed.
    pub updated: Vec<SceneIdPassEntry>,
    /// Objects that must be removed from the renderer's ID-pass draw list.
    pub removed: Vec<SceneObjectId>,
    /// Number of objects that did not change.
    pub unchanged: usize,
}

impl SceneIdPassDelta {
    fn empty() -> Self {
        Self {
            added: Vec::new(),
            updated: Vec::new(),
            removed: Vec::new(),
            unchanged: 0,
        }
    }

    /// Returns whether the commit changed any renderer entry.
    #[must_use]
    pub fn changed(&self) -> bool {
        !self.added.is_empty() || !self.updated.is_empty() || !self.removed.is_empty()
    }
}

/// The result of committing one scene submission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneIdPassCommit {
    /// Scene generation captured by a renderer frame.
    pub generation: u64,
    /// Incremental changes since the previous committed scene.
    pub delta: SceneIdPassDelta,
}

/// Persistent mapping from full scene identities to compact GPU tokens.
///
/// The GPU stores a token, while this table preserves the complete entity,
/// mesh, and subobject identity (128/128/64 bits respectively). Tokens are
/// never reused by this table, including after removal and re-addition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneIdEncoder {
    by_object: BTreeMap<SceneObjectId, u64>,
    by_token: BTreeMap<u64, SceneObjectId>,
    next_token: u64,
}

impl Default for SceneIdEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneIdEncoder {
    /// Creates an empty persistent ID mapping.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            by_object: BTreeMap::new(),
            by_token: BTreeMap::new(),
            next_token: 1,
        }
    }

    /// Returns the number of identities allocated so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_object.len()
    }

    /// Returns whether no identity has been allocated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_object.is_empty()
    }

    /// Returns a stable encoded ID, allocating a token only for a new object.
    pub fn encode(&mut self, object: SceneObjectId) -> Result<EncodedPickId, SceneIdPassError> {
        if let Some(token) = self.by_object.get(&object).copied() {
            return EncodedPickId::from_token(token).map_err(|_| SceneIdPassError::TokenExhausted);
        }
        let token = self.next_token;
        self.next_token = self
            .next_token
            .checked_add(1)
            .ok_or(SceneIdPassError::TokenExhausted)?;
        let encoded =
            EncodedPickId::from_token(token).map_err(|_| SceneIdPassError::TokenExhausted)?;
        self.by_object.insert(object, token);
        self.by_token.insert(token, object);
        Ok(encoded)
    }

    /// Resolves a GPU ID back to the complete scene identity.
    #[must_use]
    pub fn decode(&self, encoded: EncodedPickId) -> Option<SceneObjectId> {
        encoded
            .decode_token()
            .and_then(|token| self.by_token.get(&token).copied())
    }
}

/// Persistent incremental scene state used by an ID-pass renderer adapter.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SceneIdPass {
    encoder: SceneIdEncoder,
    entries: BTreeMap<SceneObjectId, SceneIdPassEntry>,
    generation: u64,
}

impl SceneIdPass {
    /// Creates an empty scene at generation zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            encoder: SceneIdEncoder::new(),
            entries: BTreeMap::new(),
            generation: 0,
        }
    }

    /// Returns the current scene generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Advances the scene generation when geometry or transforms change without
    /// changing draw handles. Existing pick/box results then become stale.
    /// Does not change tokens or require a draw-list delta.
    pub fn invalidate_content(&mut self) -> Result<u64, SceneIdPassError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(SceneIdPassError::GenerationExhausted)?;
        Ok(self.generation)
    }

    /// Returns the persistent ID encoder for renderer-side ID resolution.
    #[must_use]
    pub const fn encoder(&self) -> &SceneIdEncoder {
        &self.encoder
    }

    /// Returns the current entries in stable scene-object order.
    pub fn entries(&self) -> impl Iterator<Item = &SceneIdPassEntry> {
        self.entries.values()
    }

    /// Returns visible draws sorted by renderer key, then full object identity.
    #[must_use]
    pub fn draw_entries(&self) -> Vec<SceneIdPassEntry> {
        let mut entries: Vec<_> = self
            .entries
            .values()
            .filter(|entry| entry.item.visible)
            .copied()
            .collect();
        entries.sort_by_key(|entry| (entry.item.sort_key, entry.item.id));
        entries
    }

    /// Resolves one encoded GPU value against the current scene.
    #[must_use]
    pub fn resolve(&self, encoded: EncodedPickId) -> Option<SceneObjectId> {
        self.encoder
            .decode(encoded)
            .filter(|id| self.entries.get(id).is_some_and(|entry| entry.item.visible))
    }

    /// Resolves and deduplicates a box-selection result in stable object order.
    #[must_use]
    pub fn resolve_selection<I>(&self, ids: I) -> Vec<SceneObjectId>
    where
        I: IntoIterator<Item = EncodedPickId>,
    {
        ids.into_iter()
            .filter_map(|id| self.resolve(id))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Sends one deterministic incremental commit to a renderer adapter.
    ///
    /// The sink sees additions, updates, and removals in stable object order,
    /// so it can update a retained draw list without rebuilding the scene.
    pub fn record_commit<S>(&self, commit: &SceneIdPassCommit, sink: &mut S) -> Result<(), S::Error>
    where
        S: SceneIdPassSink,
    {
        for entry in &commit.delta.added {
            sink.add(*entry)?;
        }
        for entry in &commit.delta.updated {
            sink.update(*entry)?;
        }
        for id in &commit.delta.removed {
            sink.remove(*id)?;
        }
        Ok(())
    }

    /// Replaces the scene with a snapshot and emits only changed entries.
    pub fn submit<I>(&mut self, items: I) -> Result<SceneIdPassCommit, SceneIdPassError>
    where
        I: IntoIterator<Item = SceneIdPassItem>,
    {
        let mut desired = BTreeMap::new();
        for item in items {
            if desired.insert(item.id, item).is_some() {
                return Err(SceneIdPassError::DuplicateObject(item.id));
            }
        }
        let mut normalized: BTreeMap<_, _> = desired
            .into_iter()
            .map(|(id, item)| (id, Some(item)))
            .collect();
        for id in self.entries.keys() {
            normalized.entry(*id).or_insert(None);
        }
        self.apply_normalized(normalized)
    }

    /// Applies an incremental patch without rebuilding the scene or cloning its tables.
    /// Duplicate keys, token exhaustion, and generation overflow leave all state unchanged.
    pub fn apply_patch<I>(&mut self, operations: I) -> Result<SceneIdPassCommit, SceneIdPassError>
    where
        I: IntoIterator<Item = SceneIdPassOperation>,
    {
        let mut normalized = BTreeMap::new();
        for operation in operations {
            let (id, value) = match operation {
                SceneIdPassOperation::Upsert(item) => (item.id, Some(item)),
                SceneIdPassOperation::Remove(id) => (id, None),
            };
            if normalized.insert(id, value).is_some() {
                return Err(SceneIdPassError::DuplicatePatchKey(id));
            }
        }
        self.apply_normalized(normalized)
    }

    fn apply_normalized(
        &mut self,
        normalized: BTreeMap<SceneObjectId, Option<SceneIdPassItem>>,
    ) -> Result<SceneIdPassCommit, SceneIdPassError> {
        let changed = normalized
            .iter()
            .any(|(id, item)| self.entries.get(id).map(|entry| entry.item) != *item);
        let generation = if changed {
            self.generation
                .checked_add(1)
                .ok_or(SceneIdPassError::GenerationExhausted)?
        } else {
            self.generation
        };
        let new_tokens = normalized
            .iter()
            .filter(|(id, item)| item.is_some() && !self.encoder.by_object.contains_key(id))
            .count();
        let new_tokens = u64::try_from(new_tokens).map_err(|_| SceneIdPassError::TokenExhausted)?;
        self.encoder
            .next_token
            .checked_add(new_tokens)
            .ok_or(SceneIdPassError::TokenExhausted)?;

        let mut delta = SceneIdPassDelta::empty();
        for (id, item) in normalized {
            match item {
                Some(item) => {
                    let encoded = self.encoder.encode(id)?;
                    let entry = SceneIdPassEntry { item, encoded };
                    match self.entries.get(&id) {
                        None => delta.added.push(entry),
                        Some(previous) if previous.item != item => delta.updated.push(entry),
                        Some(_) => delta.unchanged += 1,
                    }
                    self.entries.insert(id, entry);
                }
                None => {
                    if self.entries.remove(&id).is_some() {
                        delta.removed.push(id);
                    } else {
                        delta.unchanged += 1;
                    }
                }
            }
        }
        self.generation = generation;
        Ok(SceneIdPassCommit {
            generation: self.generation,
            delta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(entity: u64, mesh: u64, subobject: u64) -> SceneObjectId {
        SceneObjectId::new(
            EntityId::from_raw(u128::from(entity)),
            MeshId::from_raw(u128::from(mesh)),
            SubobjectId::from_raw(subobject),
        )
    }

    fn item(entity: u64, mesh: u64, subobject: u64, draw_key: u64) -> SceneIdPassItem {
        SceneIdPassItem::new(
            object(entity, mesh, subobject),
            draw_key,
            draw_key + 10,
            true,
            draw_key,
        )
    }

    #[test]
    fn encoder_preserves_full_identity_and_token_stability() {
        let mut encoder = SceneIdEncoder::new();
        let id = SceneObjectId::new(
            EntityId::from_raw(u128::MAX),
            MeshId::from_raw(u128::MAX - 17),
            SubobjectId::from_raw(u64::MAX),
        );
        let first = encoder.encode(id).expect("first token");
        let second = encoder.encode(id).expect("stable token");
        assert_eq!(first, second);
        assert_eq!(encoder.decode(first), Some(id));
        assert_eq!(encoder.decode(EncodedPickId::none()), None);
        assert_eq!(
            encoder.decode(EncodedPickId::from_channels([1, 0, 1, 0])),
            None
        );
    }

    #[test]
    fn rejected_submissions_preserve_scene_and_encoder_atomically() {
        let mut pass = SceneIdPass::new();
        let a = item(1, 1, 0, 10);
        let b = item(2, 1, 0, 20);
        pass.submit([a]).expect("initial");
        let initial = pass.clone();
        assert_eq!(
            pass.submit([b, b]),
            Err(SceneIdPassError::DuplicateObject(b.id))
        );
        assert_eq!(pass, initial);
        assert_eq!(
            pass.apply_patch([
                SceneIdPassOperation::Upsert(b),
                SceneIdPassOperation::Remove(b.id)
            ]),
            Err(SceneIdPassError::DuplicatePatchKey(b.id))
        );
        assert_eq!(pass, initial);

        pass.encoder.next_token = u64::MAX;
        let exhausted = pass.clone();
        assert_eq!(
            pass.apply_patch([
                SceneIdPassOperation::Remove(a.id),
                SceneIdPassOperation::Upsert(b)
            ]),
            Err(SceneIdPassError::TokenExhausted)
        );
        assert_eq!(pass, exhausted);
        pass.generation = u64::MAX;
        let exhausted = pass.clone();
        assert_eq!(pass.submit([]), Err(SceneIdPassError::GenerationExhausted));
        assert_eq!(pass, exhausted);
        assert_eq!(
            pass.submit([a]).expect("unchanged at maximum").generation,
            u64::MAX
        );
    }

    #[test]
    fn visible_draw_order_and_readded_identity_remain_stable() {
        let mut pass = SceneIdPass::new();
        let a = item(1, 1, 0, 30);
        let b = item(2, 1, 0, 10);
        let hidden = SceneIdPassItem {
            visible: false,
            ..item(3, 1, 0, 1)
        };
        pass.submit([a, b, hidden]).expect("snapshot");
        let draws = pass.draw_entries();
        assert_eq!(
            draws
                .iter()
                .map(|entry| entry.item().id)
                .collect::<Vec<_>>(),
            vec![b.id, a.id]
        );
        let encoded = draws[1].encoded_id();
        let hidden_encoded = pass
            .entries()
            .find(|entry| entry.item().id == hidden.id)
            .expect("hidden entry")
            .encoded_id();
        assert_eq!(pass.resolve(hidden_encoded), None);
        pass.apply_patch([SceneIdPassOperation::Remove(a.id)])
            .expect("remove");
        assert_eq!(pass.resolve(encoded), None);
        let commit = pass
            .apply_patch([SceneIdPassOperation::Upsert(a)])
            .expect("readd");
        assert_eq!(commit.delta.added[0].encoded_id(), encoded);
        assert_eq!(pass.resolve(encoded), Some(a.id));
    }

    #[test]
    fn snapshot_submission_is_incremental_and_deterministic() {
        let mut pass = SceneIdPass::new();
        let first = pass
            .submit([item(2, 1, 0, 20), item(1, 1, 0, 10)])
            .expect("first");
        assert_eq!(first.generation, 1);
        assert_eq!(first.delta.added.len(), 2);
        let unchanged = pass
            .submit([item(1, 1, 0, 10), item(2, 1, 0, 20)])
            .expect("same");
        assert_eq!(unchanged.generation, 1);
        assert_eq!(unchanged.delta.unchanged, 2);
        let changed = pass
            .submit([item(1, 1, 0, 11), item(3, 2, 4, 30)])
            .expect("changed");
        assert_eq!(changed.generation, 2);
        assert_eq!(changed.delta.updated.len(), 1);
        assert_eq!(changed.delta.added.len(), 1);
        assert_eq!(changed.delta.removed, vec![object(2, 1, 0)]);
    }

    #[test]
    fn patch_and_box_resolution_return_only_current_objects() {
        let mut pass = SceneIdPass::new();
        let a = item(1, 2, 3, 10);
        let b = item(2, 2, 3, 20);
        pass.apply_patch([
            SceneIdPassOperation::Upsert(a),
            SceneIdPassOperation::Upsert(b),
        ])
        .expect("patch");
        let encoded_a = pass
            .entries()
            .find(|entry| entry.item().id == a.id)
            .expect("a")
            .encoded_id();
        let encoded_b = pass
            .entries()
            .find(|entry| entry.item().id == b.id)
            .expect("b")
            .encoded_id();
        assert_eq!(
            pass.resolve_selection([encoded_b, encoded_a, encoded_a]),
            vec![a.id, b.id]
        );
        pass.apply_patch([SceneIdPassOperation::Remove(a.id)])
            .expect("remove");
        assert_eq!(pass.resolve(encoded_a), None);
        assert_eq!(pass.resolve(encoded_b), Some(b.id));
    }

    #[derive(Default)]
    struct RecordedOperations {
        operations: Vec<(char, SceneObjectId, Option<EncodedPickId>)>,
    }

    impl SceneIdPassSink for RecordedOperations {
        type Error = core::convert::Infallible;

        fn add(&mut self, entry: SceneIdPassEntry) -> Result<(), Self::Error> {
            self.operations
                .push(('a', entry.item().id, Some(entry.encoded_id())));
            Ok(())
        }

        fn update(&mut self, entry: SceneIdPassEntry) -> Result<(), Self::Error> {
            self.operations
                .push(('u', entry.item().id, Some(entry.encoded_id())));
            Ok(())
        }

        fn remove(&mut self, id: SceneObjectId) -> Result<(), Self::Error> {
            self.operations.push(('r', id, None));
            Ok(())
        }
    }

    #[test]
    fn renderer_sink_receives_stable_incremental_entries() {
        let mut pass = SceneIdPass::new();
        let initial = pass
            .submit([item(2, 1, 0, 20), item(1, 1, 0, 10)])
            .expect("initial");
        let mut sink = RecordedOperations::default();
        pass.record_commit(&initial, &mut sink)
            .expect("record initial");
        assert_eq!(sink.operations.len(), 2);
        assert_eq!(sink.operations[0].0, 'a');
        assert_eq!(sink.operations[0].1, object(1, 1, 0));
        assert_eq!(sink.operations[1].1, object(2, 1, 0));
        let first_encoded = sink.operations[0].2.expect("encoded");

        let changed = pass
            .submit([item(1, 1, 0, 11)])
            .expect("incremental change");
        sink.operations.clear();
        pass.record_commit(&changed, &mut sink)
            .expect("record change");
        assert_eq!(sink.operations.len(), 2);
        assert_eq!(sink.operations[0].0, 'u');
        assert_eq!(sink.operations[0].2, Some(first_encoded));
        assert_eq!(sink.operations[1], ('r', object(2, 1, 0), None));
    }
}
