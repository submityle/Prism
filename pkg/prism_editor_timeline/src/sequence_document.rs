//! Owned sequence projects, deterministic files and inverse-only authoring history.
use crate::{
    PlaybackRate, SequenceClip, SequenceClipId, SequenceError, SequenceId, SequenceLimits,
    SequenceRange, SequenceRegistry, TimelineCommand, TimelineDocument, TimelineEdit,
    TimelineError,
};
use alloc::{
    collections::{BTreeMap, VecDeque},
    string::String,
    sync::Arc,
    vec::Vec,
};
use core::fmt;

const MAGIC: &[u8; 8] = b"PRSSEQ01";

/// Project and authoring resource policy, independent of the serialized file.
#[derive(Clone, Copy, Debug)]
pub struct SequenceDocumentLimits {
    /// Registry and per-document limits.
    pub sequence: SequenceLimits,
    /// Entire project file, at most 64 MiB.
    pub max_bytes: usize,
    /// Atomic command edits, at most 4096.
    pub max_edits: usize,
    /// Retained undo plus redo records, at most 4096.
    pub max_history_entries: usize,
    /// Logical retained inverse/forward payload reservation, at most 64 MiB.
    pub max_history_bytes: usize,
    /// Command label bytes, at most 4096.
    pub max_label_bytes: usize,
}
impl Default for SequenceDocumentLimits {
    fn default() -> Self {
        Self {
            sequence: SequenceLimits::default(),
            max_bytes: 16 * 1024 * 1024,
            max_edits: 1024,
            max_history_entries: 256,
            max_history_bytes: 16 * 1024 * 1024,
            max_label_bytes: 4096,
        }
    }
}
impl SequenceDocumentLimits {
    fn check(self) -> Result<(), SequenceDocumentError> {
        SequenceRegistry::new(self.sequence)?;
        if self.max_bytes == 0
            || self.max_bytes > 64 * 1024 * 1024
            || self.max_edits == 0
            || self.max_edits > 4096
            || self.max_history_entries == 0
            || self.max_history_entries > 4096
            || self.max_history_bytes == 0
            || self.max_history_bytes > 64 * 1024 * 1024
            || self.max_label_bytes == 0
            || self.max_label_bytes > 4096
        {
            return Err(SequenceDocumentError::InvalidLimits);
        }
        Ok(())
    }
}

/// Exact project validation or authoring failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SequenceDocumentError {
    /// Invalid caller resource policy.
    InvalidLimits,
    /// Malformed, noncanonical, truncated or trailing bytes.
    InvalidEncoding,
    /// Named resource cap exceeded before adoption.
    Limit(&'static str),
    /// Missing clip being removed.
    MissingClip(SequenceClipId),
    /// High-water identity or revision cannot advance.
    Exhausted,
    /// High-water metadata is below a live identity.
    InvalidHighWater,
    /// Empty command or label.
    EmptyCommand,
    /// Real nested graph validation failure.
    Sequence(SequenceError),
    /// Real timeline codec failure.
    Timeline(TimelineError),
}
impl From<SequenceError> for SequenceDocumentError {
    fn from(value: SequenceError) -> Self {
        Self::Sequence(value)
    }
}
impl From<TimelineError> for SequenceDocumentError {
    fn from(value: TimelineError) -> Self {
        Self::Timeline(value)
    }
}
impl fmt::Display for SequenceDocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sequence document {self:?}")
    }
}
impl core::error::Error for SequenceDocumentError {}

/// Complete owned project. Private maps prevent bypassing topology/history checks.
/// Zero is a valid identity; absent high-water means no ID has ever been used.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SequenceProjectDocument {
    documents: BTreeMap<SequenceId, TimelineDocument>,
    clips: BTreeMap<SequenceClipId, (SequenceId, SequenceClip)>,
    root: Option<SequenceId>,
    sequence_high_water: Option<u128>,
    clip_high_water: Option<u128>,
}
impl SequenceProjectDocument {
    /// Project root. An empty project has no root.
    pub const fn root(&self) -> Option<SequenceId> {
        self.root
    }
    /// Original owned document.
    pub fn document(&self, id: SequenceId) -> Option<&TimelineDocument> {
        self.documents.get(&id)
    }
    /// Stable sequence identity order.
    pub fn documents(&self) -> impl Iterator<Item = (SequenceId, &TimelineDocument)> {
        self.documents.iter().map(|(id, document)| (*id, document))
    }
    /// Global stable clip identity order with owning parent.
    pub fn clips(&self) -> impl Iterator<Item = (SequenceId, &SequenceClip)> {
        self.clips.values().map(|(parent, clip)| (*parent, clip))
    }
    /// Highest sequence ID ever adopted, including removed/undone entries.
    pub const fn sequence_high_water(&self) -> Option<u128> {
        self.sequence_high_water
    }
    /// Highest clip ID ever adopted, including removed/undone entries.
    pub const fn clip_high_water(&self) -> Option<u128> {
        self.clip_high_water
    }
    /// Suggested fresh sequence ID; exhaustion never wraps to a reused ID.
    pub fn next_sequence_id(&self) -> Result<SequenceId, SequenceDocumentError> {
        Ok(SequenceId(next_id(self.sequence_high_water)?))
    }
    /// Suggested fresh clip ID; allocation occurs on successful command adoption.
    pub fn next_clip_id(&self) -> Result<SequenceClipId, SequenceDocumentError> {
        Ok(SequenceClipId(next_id(self.clip_high_water)?))
    }
    /// Borrows the actual owned documents into the real sampling/event registry.
    pub fn registry(
        &self,
        limits: SequenceLimits,
    ) -> Result<SequenceRegistry<'_>, SequenceDocumentError> {
        let mut registry = SequenceRegistry::new(limits)?;
        for (id, document) in &self.documents {
            registry.insert(*id, document)?;
        }
        for (parent, clip) in self.clips.values() {
            registry.add_clip(*parent, clip.clone())?;
        }
        if let Some(root) = self.root
            && !self.documents.contains_key(&root)
        {
            return Err(SequenceError::MissingSequence(root).into());
        }
        if self
            .documents
            .keys()
            .any(|id| Some(id.0) > self.sequence_high_water)
            || self
                .clips
                .keys()
                .any(|id| Some(id.0) > self.clip_high_water)
        {
            return Err(SequenceDocumentError::InvalidHighWater);
        }
        Ok(registry)
    }
    /// Deterministic versioned little-endian file, preserving real document bits.
    pub fn encode(&self, limits: SequenceDocumentLimits) -> Result<Vec<u8>, SequenceDocumentError> {
        limits.check()?;
        self.registry(limits.sequence)?;
        let mut out = Output {
            bytes: Vec::new(),
            max: limits.max_bytes,
        };
        out.put(MAGIC)?;
        out.optional_id(self.sequence_high_water)?;
        out.optional_id(self.clip_high_water)?;
        out.optional_id(self.root.map(|id| id.0))?;
        out.u64(self.documents.len() as u64)?;
        for (id, document) in &self.documents {
            out.put(&id.0.to_le_bytes())?;
            let bytes =
                document.encode(limits.sequence.document, limits.sequence.max_document_bytes)?;
            out.u64(bytes.len() as u64)?;
            out.put(&bytes)?;
        }
        out.u64(self.clips.len() as u64)?;
        for (parent, clip) in self.clips.values() {
            out.clip(*parent, clip)?;
        }
        Ok(out.bytes)
    }
    /// Checks every count against both policy and remaining bytes before allocating.
    /// No host project is mutated; duplicates, cycles and missing references reject.
    pub fn decode(
        bytes: &[u8],
        limits: SequenceDocumentLimits,
    ) -> Result<Self, SequenceDocumentError> {
        limits.check()?;
        bound(bytes.len(), limits.max_bytes, "project bytes")?;
        let mut input = Input(bytes);
        if &input.array::<8>()? != MAGIC {
            return Err(SequenceDocumentError::InvalidEncoding);
        }
        let mut project = Self {
            sequence_high_water: input.optional_id()?,
            clip_high_water: input.optional_id()?,
            root: input.optional_id()?.map(SequenceId),
            ..Self::default()
        };
        let count = input.count(limits.sequence.max_sequences, 24, "sequences")?;
        let mut previous = None;
        for _ in 0..count {
            let id = SequenceId(input.id()?);
            ordered(&mut previous, id)?;
            let len = input.count(limits.sequence.max_document_bytes, 1, "document bytes")?;
            let document = TimelineDocument::decode(
                input.take(len)?,
                limits.sequence.document,
                limits.sequence.max_document_bytes,
            )?;
            project.documents.insert(id, document);
        }
        let count = input.count(limits.sequence.max_clips, 104, "clips")?;
        let mut previous = None;
        for _ in 0..count {
            let (parent, clip) = input.clip()?;
            ordered(&mut previous, clip.id)?;
            project.clips.insert(clip.id, (parent, clip));
        }
        if !input.0.is_empty() {
            return Err(SequenceDocumentError::InvalidEncoding);
        }
        project.registry(limits.sequence)?;
        Ok(project)
    }
}
fn next_id(high: Option<u128>) -> Result<u128, SequenceDocumentError> {
    match high {
        None => Ok(1),
        Some(id) => id.checked_add(1).ok_or(SequenceDocumentError::Exhausted),
    }
}

/// One authoring edit. Replacement history retains only the affected document.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceProjectEdit {
    /// Insert or replace an owned real document under stable identity.
    PutDocument {
        /// Stable document ID.
        id: SequenceId,
        /// Actual authored timeline and clock.
        document: TimelineDocument,
    },
    /// Remove a document. References must be removed in the same transaction.
    RemoveDocument(SequenceId),
    /// Insert or replace/move a globally identified clip.
    PutClip {
        /// Owning sequence.
        parent: SequenceId,
        /// Complete authored clip.
        clip: SequenceClip,
    },
    /// Remove one clip.
    RemoveClip(SequenceClipId),
    /// Choose or clear the root.
    SetRoot(Option<SequenceId>),
    /// Change one document clock, validating all affected clip mappings.
    SetFrequency {
        /// Existing sequence.
        id: SequenceId,
        /// Positive integer frequency.
        ticks_per_second: u64,
    },
    /// Edit a real track/key/event, retaining only the actual inverse payload.
    Timeline {
        /// Existing sequence.
        id: SequenceId,
        /// Actual kernel edit.
        edit: TimelineEdit,
    },
}
/// Explicit atomic command data, with no UI or backend policy.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceProjectCommand {
    /// Bounded nonempty host label.
    pub label: String,
    /// Ordered edits; only the final topology must be valid.
    pub edits: Vec<SequenceProjectEdit>,
}
/// Exact host identity for late save confirmation, never inferred from revision.
#[derive(Clone, Debug)]
pub struct SequenceProjectToken(Arc<()>);
impl PartialEq for SequenceProjectToken {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for SequenceProjectToken {}
struct Record {
    command: SequenceProjectCommand,
    bytes: usize,
}
/// Owns project state and bounded inverse commands. Readers only borrow state.
pub struct SequenceProjectEditor {
    document: SequenceProjectDocument,
    limits: SequenceDocumentLimits,
    revision: u64,
    token: SequenceProjectToken,
    undo: VecDeque<Record>,
    redo: Vec<Record>,
    history_bytes: usize,
}
impl SequenceProjectEditor {
    /// Validates a loaded candidate without recording a synthetic edit.
    pub fn new(
        document: SequenceProjectDocument,
        limits: SequenceDocumentLimits,
    ) -> Result<Self, SequenceDocumentError> {
        document.encode(limits)?;
        Ok(Self {
            document,
            limits,
            revision: 0,
            token: SequenceProjectToken(Arc::new(())),
            undo: VecDeque::new(),
            redo: Vec::new(),
            history_bytes: 0,
        })
    }
    /// Authoritative immutable project.
    pub const fn document(&self) -> &SequenceProjectDocument {
        &self.document
    }
    /// Resource policy.
    pub const fn limits(&self) -> SequenceDocumentLimits {
        self.limits
    }
    /// Monotonic successful-change revision, including undo/redo.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Compare against save receipts before marking live state clean.
    pub fn snapshot_token(&self) -> SequenceProjectToken {
        self.token.clone()
    }
    /// Number of available inverses.
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }
    /// Number of available replays.
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }
    /// Reserved logical payload bytes across both history directions.
    pub const fn history_bytes(&self) -> usize {
        self.history_bytes
    }
    /// Next inverse label.
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.back().map(|r| r.command.label.as_str())
    }
    /// Applies ordered edits with inverse rollback; never clones the project.
    pub fn execute(
        &mut self,
        command: SequenceProjectCommand,
    ) -> Result<(), SequenceDocumentError> {
        self.revision
            .checked_add(1)
            .ok_or(SequenceDocumentError::Exhausted)?;
        let forward = command_bytes(&command, self.limits)?;
        let old_high = (
            self.document.sequence_high_water,
            self.document.clip_high_water,
        );
        let inverse = self.apply(command)?;
        let validation = self
            .document
            .encode(self.limits)
            .and_then(|_| command_bytes(&inverse, self.limits));
        let bytes = match validation {
            Ok(bytes) => bytes.max(forward),
            Err(error) => {
                self.apply(inverse)
                    .expect("owned inverse restores transaction");
                (
                    self.document.sequence_high_water,
                    self.document.clip_high_water,
                ) = old_high;
                return Err(error);
            }
        };
        for record in self.redo.drain(..) {
            self.history_bytes -= record.bytes;
        }
        while self.undo.len() >= self.limits.max_history_entries
            || self.history_bytes > self.limits.max_history_bytes - bytes
        {
            self.history_bytes -= self.undo.pop_front().expect("history accounting").bytes;
        }
        self.history_bytes += bytes;
        self.undo.push_back(Record {
            command: inverse,
            bytes,
        });
        self.changed();
        Ok(())
    }
    /// Undo only affected payloads; successful ID high-water never decreases.
    pub fn undo(&mut self) -> Result<bool, SequenceDocumentError> {
        if self.undo.is_empty() {
            return Ok(false);
        }
        self.revision
            .checked_add(1)
            .ok_or(SequenceDocumentError::Exhausted)?;
        let record = self.undo.pop_back().expect("nonempty");
        let command = self.apply(record.command).expect("private inverse");
        if let Err(error) = self.document.encode(self.limits) {
            let command = self.apply(command).expect("undo validation rollback");
            self.undo.push_back(Record { command, ..record });
            return Err(error);
        }
        self.redo.push(Record { command, ..record });
        self.changed();
        Ok(true)
    }
    /// Reapply the last undone command without retaining any project snapshot.
    pub fn redo(&mut self) -> Result<bool, SequenceDocumentError> {
        if self.redo.is_empty() {
            return Ok(false);
        }
        self.revision
            .checked_add(1)
            .ok_or(SequenceDocumentError::Exhausted)?;
        let record = self.redo.pop().expect("nonempty");
        let command = self.apply(record.command).expect("private replay");
        if let Err(error) = self.document.encode(self.limits) {
            let command = self.apply(command).expect("redo validation rollback");
            self.redo.push(Record { command, ..record });
            return Err(error);
        }
        self.undo.push_back(Record { command, ..record });
        self.changed();
        Ok(true)
    }
    fn changed(&mut self) {
        self.revision += 1;
        self.token = SequenceProjectToken(Arc::new(()));
    }
    fn apply(
        &mut self,
        command: SequenceProjectCommand,
    ) -> Result<SequenceProjectCommand, SequenceDocumentError> {
        let high = (
            self.document.sequence_high_water,
            self.document.clip_high_water,
        );
        let mut inverse = Vec::with_capacity(command.edits.len());
        for edit in command.edits {
            match self.one(edit) {
                Ok(edit) => inverse.push(edit),
                Err(error) => {
                    for edit in inverse.into_iter().rev() {
                        self.one(edit).expect("rollback");
                    }
                    (
                        self.document.sequence_high_water,
                        self.document.clip_high_water,
                    ) = high;
                    return Err(error);
                }
            }
        }
        inverse.reverse();
        Ok(SequenceProjectCommand {
            label: command.label,
            edits: inverse,
        })
    }
    fn one(
        &mut self,
        edit: SequenceProjectEdit,
    ) -> Result<SequenceProjectEdit, SequenceDocumentError> {
        use SequenceProjectEdit::{PutClip, PutDocument, RemoveClip, RemoveDocument, SetRoot};
        Ok(match edit {
            PutDocument { id, document } => {
                self.document.sequence_high_water =
                    self.document.sequence_high_water.max(Some(id.0));
                match self.document.documents.insert(id, document) {
                    Some(document) => PutDocument { id, document },
                    None => RemoveDocument(id),
                }
            }
            RemoveDocument(id) => PutDocument {
                id,
                document: self
                    .document
                    .documents
                    .remove(&id)
                    .ok_or(SequenceError::MissingSequence(id))?,
            },
            PutClip { parent, clip } => {
                self.document.clip_high_water = self.document.clip_high_water.max(Some(clip.id.0));
                let id = clip.id;
                match self.document.clips.insert(id, (parent, clip)) {
                    Some((parent, clip)) => PutClip { parent, clip },
                    None => RemoveClip(id),
                }
            }
            RemoveClip(id) => {
                let (parent, clip) = self
                    .document
                    .clips
                    .remove(&id)
                    .ok_or(SequenceDocumentError::MissingClip(id))?;
                PutClip { parent, clip }
            }
            SetRoot(root) => SetRoot(core::mem::replace(&mut self.document.root, root)),
            SequenceProjectEdit::SetFrequency {
                id,
                ticks_per_second,
            } => {
                let document = self
                    .document
                    .documents
                    .get_mut(&id)
                    .ok_or(SequenceError::MissingSequence(id))?;
                SequenceProjectEdit::SetFrequency {
                    id,
                    ticks_per_second: core::mem::replace(
                        &mut document.ticks_per_second,
                        ticks_per_second,
                    ),
                }
            }
            SequenceProjectEdit::Timeline { id, edit } => {
                let document = self
                    .document
                    .documents
                    .get_mut(&id)
                    .ok_or(SequenceError::MissingSequence(id))?;
                SequenceProjectEdit::Timeline {
                    id,
                    edit: timeline_edit(&mut document.timeline, edit)?,
                }
            }
        })
    }
}
fn command_bytes(
    command: &SequenceProjectCommand,
    limits: SequenceDocumentLimits,
) -> Result<usize, SequenceDocumentError> {
    if command.label.trim().is_empty() || command.edits.is_empty() {
        return Err(SequenceDocumentError::EmptyCommand);
    }
    bound(command.label.len(), limits.max_label_bytes, "label bytes")?;
    bound(command.edits.len(), limits.max_edits, "command edits")?;
    let mut bytes = command.label.len();
    for edit in &command.edits {
        bytes = bytes.saturating_add(size_of::<SequenceProjectEdit>());
        if let SequenceProjectEdit::PutDocument { document, .. } = edit {
            bytes = bytes.saturating_add(
                document
                    .encode(limits.sequence.document, limits.sequence.max_document_bytes)?
                    .len(),
            );
        }
        if let SequenceProjectEdit::Timeline { edit, .. } = edit {
            let command = TimelineCommand {
                label: command.label.clone(),
                edits: alloc::vec![edit.clone()],
            };
            bytes = bytes.saturating_add(
                command
                    .encode(crate::TimelineLimits {
                        max_history_bytes: limits.max_history_bytes,
                        max_edits: 1,
                        ..limits.sequence.document
                    })?
                    .len(),
            );
        }
    }
    bound(bytes, limits.max_history_bytes, "history bytes")?;
    Ok(bytes)
}
fn timeline_edit(
    timeline: &mut crate::Timeline,
    edit: TimelineEdit,
) -> Result<TimelineEdit, TimelineError> {
    Ok(match edit {
        TimelineEdit::AddTrack(track) => {
            let id = track.id();
            timeline.add_track(track)?;
            TimelineEdit::RemoveTrack(id)
        }
        TimelineEdit::RemoveTrack(id) => TimelineEdit::AddTrack(timeline.remove_track(id)?),
        TimelineEdit::PutKey { track, key } => {
            let target = timeline
                .track_mut(track)
                .ok_or(TimelineError::MissingTrack(track))?;
            if target.key(key.id()).is_some() {
                TimelineEdit::PutKey {
                    track,
                    key: target.replace(key)?,
                }
            } else {
                let id = key.id();
                target.insert(key)?;
                TimelineEdit::RemoveKey { track, key: id }
            }
        }
        TimelineEdit::RemoveKey { track, key } => TimelineEdit::PutKey {
            track,
            key: timeline
                .track_mut(track)
                .ok_or(TimelineError::MissingTrack(track))?
                .remove(key)?,
        },
        TimelineEdit::AddEvent(event) => {
            let id = event.id;
            timeline.add_event(event)?;
            TimelineEdit::RemoveEvent(id)
        }
        TimelineEdit::RemoveEvent(id) => TimelineEdit::AddEvent(timeline.remove_event(id)?),
    })
}
fn bound(actual: usize, max: usize, resource: &'static str) -> Result<(), SequenceDocumentError> {
    if actual > max {
        Err(SequenceDocumentError::Limit(resource))
    } else {
        Ok(())
    }
}
fn ordered<T: Ord + Copy>(previous: &mut Option<T>, next: T) -> Result<(), SequenceDocumentError> {
    if previous.is_some_and(|value| value >= next) {
        return Err(SequenceDocumentError::InvalidEncoding);
    }
    *previous = Some(next);
    Ok(())
}
struct Output {
    bytes: Vec<u8>,
    max: usize,
}
impl Output {
    fn put(&mut self, bytes: &[u8]) -> Result<(), SequenceDocumentError> {
        bound(
            bytes.len(),
            self.max.saturating_sub(self.bytes.len()),
            "project bytes",
        )?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn u64(&mut self, value: u64) -> Result<(), SequenceDocumentError> {
        self.put(&value.to_le_bytes())
    }
    fn optional_id(&mut self, value: Option<u128>) -> Result<(), SequenceDocumentError> {
        self.put(&[u8::from(value.is_some())])?;
        self.put(&value.unwrap_or(0).to_le_bytes())
    }
    fn clip(
        &mut self,
        parent: SequenceId,
        clip: &SequenceClip,
    ) -> Result<(), SequenceDocumentError> {
        for id in [clip.id.0, parent.0, clip.source.0] {
            self.put(&id.to_le_bytes())?;
        }
        let (n, d) = clip.rate.ratio();
        for value in [
            clip.parent.start(),
            clip.parent.end(),
            clip.source_range.start(),
            clip.source_range.end(),
            n,
            d,
            clip.repetitions,
        ] {
            self.u64(value)?;
        }
        Ok(())
    }
}
struct Input<'a>(&'a [u8]);
impl<'a> Input<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], SequenceDocumentError> {
        if len > self.0.len() {
            return Err(SequenceDocumentError::InvalidEncoding);
        }
        let (head, rest) = self.0.split_at(len);
        self.0 = rest;
        Ok(head)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], SequenceDocumentError> {
        Ok(self.take(N)?.try_into().expect("exact slice"))
    }
    fn u64(&mut self) -> Result<u64, SequenceDocumentError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn id(&mut self) -> Result<u128, SequenceDocumentError> {
        Ok(u128::from_le_bytes(self.array()?))
    }
    fn optional_id(&mut self) -> Result<Option<u128>, SequenceDocumentError> {
        let flag = self.array::<1>()?[0];
        let id = self.id()?;
        match flag {
            0 if id == 0 => Ok(None),
            1 => Ok(Some(id)),
            _ => Err(SequenceDocumentError::InvalidEncoding),
        }
    }
    fn count(
        &mut self,
        max: usize,
        minimum: usize,
        resource: &'static str,
    ) -> Result<usize, SequenceDocumentError> {
        let count =
            usize::try_from(self.u64()?).map_err(|_| SequenceDocumentError::Limit(resource))?;
        bound(count, max, resource)?;
        if count > self.0.len() / minimum {
            return Err(SequenceDocumentError::InvalidEncoding);
        }
        Ok(count)
    }
    fn clip(&mut self) -> Result<(SequenceId, SequenceClip), SequenceDocumentError> {
        let id = SequenceClipId(self.id()?);
        let parent = SequenceId(self.id()?);
        let source = SequenceId(self.id()?);
        let parent_range = SequenceRange::new(self.u64()?, self.u64()?)?;
        let source_range = SequenceRange::new(self.u64()?, self.u64()?)?;
        let (n, d) = (self.u64()?, self.u64()?);
        let rate = PlaybackRate::new(n, d).map_err(|_| SequenceDocumentError::InvalidEncoding)?;
        if rate.ratio() != (n, d) {
            return Err(SequenceDocumentError::InvalidEncoding);
        }
        Ok((
            parent,
            SequenceClip {
                id,
                source,
                parent: parent_range,
                source_range,
                rate,
                repetitions: self.u64()?,
            },
        ))
    }
}
