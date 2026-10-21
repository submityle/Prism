//! Generic authoring documents and asynchronous, revision-fenced cell storage.
//!
//! `V1` is `PRISMDOC`, `u32` version, `u64` source model revision, `u32` object
//! count, `u32` row count, followed by objects and parent-before-child
//! `Outliner` rows. Integers are little endian; strings/binaries have `u32`
//! byte lengths. Object records contain the complete tagged `SelectionKind`,
//! registered type path,
//! caller-assigned schema version, and Prism reflection binary. Rows contain
//! `u128` id, optional `u128` parent, `UTF-8` label, optional selection identity,
//! and expanded/visible booleans. Child lists are derived, never authoritative.
//!
//! The caller owns schema versions and registry factories, including nested
//! registrations and validation metadata. A type-path/schema change requires a
//! new version; there are no implicit migrations or business-type assumptions.
//! Selection, command callbacks, undo history and transient UI state are not
//! persisted. A restored controller begins a new, clean model-revision session.
//!
//! Native integration: construct `FileCellIoProvider` once outside the frame
//! loop, then `DocumentFileAdapter` with a bounded `CellLoadRequest` supplied by
//! the world owner. `submit_save`/`submit_load` and `poll` never do filesystem
//! I/O. They reuse `WorldIoDriver` and the production provider's locks, envelope
//! checks,
//! atomic replacement and durability policy. Snapshot encoding and receipt
//! preparation are bounded CPU work, not constant-time operations; applications
//! with large documents should prepare/load receipts on a worker. Poll registers
//! the supplied waker, suitable for a native event-loop proxy.
//!
//! A save receipt only clears dirty state after both `model_revision` and a full
//! snapshot match. A load receipt first builds an entirely validated candidate;
//! build replacement UI from `PreparedDocumentLoad::controller()`, then call
//! `install` and publish that UI together on the native thread. Failed decoding,
//! stale receipts and rejected schemas leave the old controller/UI untouched.
//! Do not discard the old UI before install succeeds.
//!
//! Disk revisions are independent of controller revisions: new files use zero
//! (therefore cannot overwrite any existing differing revision); loaded files
//! use base+1 with compare-and-swap. Failed/cancelled saves require an identical
//! retry or a load/install before another snapshot may be saved. Cancellation
//! cannot undo an already-published commit. A failed load is not proof of an
//! absent file and never resets the known durable base.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::String,
    sync::Arc,
    vec::Vec,
};
use core::{
    fmt,
    task::{Context, Poll},
};

use prism_editor_core::{SelectionItem, SelectionKind};
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_world::{
    CellLoadRequest, CellSaveRequest, FileCellIoError, FileCellIoProvider, IoBudget, IoCompletion,
    IoFailure, IoRequestId, IoSubmitError, WorldIoDriver,
};
use prism_reflect::{
    from_binary_with_limits, to_binary, DeserializeLimits, Reflect, ReflectRef, TypeRegistry,
};

use crate::{ControllerError, ControllerLimits, DesktopUiError, EditorController, EditorDesktopUi};

const MAGIC: &[u8; 8] = b"PRISMDOC";
const VERSION: u32 = 1;

/// Explicit bounded policy for both trusted snapshots and untrusted documents.
#[derive(Clone, Copy, Debug)]
pub struct DocumentLimits {
    /// Maximum complete document payload bytes, excluding the cell envelope.
    pub max_document_bytes: usize,
    /// Maximum reflection binary and conservative allocation estimate per object.
    pub max_object_bytes: usize,
    /// Maximum objects, including those absent from the Outliner and selection.
    pub max_objects: usize,
    /// Maximum Outliner rows, including hidden and collapsed descendants.
    pub max_outliner_nodes: usize,
    /// Maximum UTF-8 bytes in any identity domain, type path, label or model string.
    pub max_string_bytes: usize,
    /// Maximum reflected nodes across all document objects, counting map keys.
    pub max_reflect_nodes: usize,
    /// Maximum reflected depth; the root counts as depth one.
    pub max_reflect_depth: usize,
    /// Maximum elements in any list, array, map or set.
    pub max_collection_len: usize,
}

impl Default for DocumentLimits {
    fn default() -> Self {
        Self {
            max_document_bytes: 8 * 1024 * 1024,
            max_object_bytes: 1024 * 1024,
            max_objects: 4096,
            max_outliner_nodes: 4096,
            max_string_bytes: 4096,
            max_reflect_nodes: 131_072,
            max_reflect_depth: 64,
            max_collection_len: 65_536,
        }
    }
}

impl DocumentLimits {
    fn validate(self) -> Result<(), DocumentError> {
        if self.max_document_bytes < 28
            || self.max_document_bytes > 64 * 1024 * 1024 - 64
            || self.max_object_bytes > self.max_document_bytes
            || self.max_string_bytes > self.max_document_bytes
            || self.max_reflect_depth == 0
            || self.max_reflect_depth > 128
        {
            return Err(DocumentError::InvalidLimits);
        }
        Ok(())
    }
}

/// Codec, optimistic concurrency, or submission error. Storage failures are
/// delivered separately in [`DocumentIoEvent::Failed`], preserving their type.
#[derive(Debug)]
pub enum DocumentError {
    /// Invalid or inconsistent codec/provider resource policy.
    InvalidLimits,
    /// A named document resource budget was exceeded.
    Limit(&'static str),
    /// Invalid framing, identity, hierarchy, `UTF-8`, version or trailing data.
    BadFormat(&'static str),
    /// The caller did not allowlist this type path and schema version.
    UnknownSchema {
        /// Persisted root type path.
        type_path: String,
        /// Requested schema version; zero is also used for an unlisted encoder root.
        version: u32,
    },
    /// A schema path was registered more than once.
    DuplicateSchema(String),
    /// A complete typed object identity occurred more than once.
    DuplicateObject(SelectionItem),
    /// An `Outliner` row id occurred more than once.
    DuplicateOutliner(OutlinerId),
    /// The factory or controller registry is missing the declared root type.
    Registry(String),
    /// `Prism` reflection serialization or deserialization rejected an object.
    Reflection(String),
    /// Controller validation or save acknowledgement rejected the operation.
    Controller(ControllerError),
    /// Candidate retained-UI construction or save acknowledgement failed.
    Ui(DesktopUiError),
    /// An accepted document operation has not yet delivered its terminal event.
    Busy,
    /// Native document replacement was requested while the live model is dirty.
    UnsavedChanges,
    /// An uncertain/failed save requires identical retry or explicit reload/install.
    ReconciliationRequired,
    /// No failed or cancelled save is retained for identical retry.
    NoRetry,
    /// A durable revision or adapter operation generation would overflow.
    RevisionExhausted,
    /// Model edits occurred after the receipt's snapshot was captured.
    StaleRevision {
        /// Model revision captured at submission.
        saved: u64,
        /// Current live controller model revision.
        current: u64,
    },
    /// Content changed without advancing the model revision, such as registration.
    SnapshotChanged,
    /// A load receipt is for another adapter or has been superseded by later I/O.
    SupersededReceipt,
    /// The world driver or provider rejected admission; no request was accepted.
    Submit(IoSubmitError<FileCellIoError>),
}

impl fmt::Display for DocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => f.write_str("invalid document limits"),
            Self::Limit(name) => write!(f, "document limit exceeded: {name}"),
            Self::BadFormat(reason) => write!(f, "invalid document: {reason}"),
            Self::UnknownSchema { type_path, version } => {
                write!(f, "unknown document schema {type_path}@{version}")
            }
            Self::DuplicateSchema(name) => write!(f, "duplicate schema: {name}"),
            Self::DuplicateObject(item) => write!(f, "duplicate document object: {item:?}"),
            Self::DuplicateOutliner(id) => write!(f, "duplicate Outliner row: {id:?}"),
            Self::Registry(reason) => write!(f, "document registry: {reason}"),
            Self::Reflection(reason) => write!(f, "document reflection binary: {reason}"),
            Self::Controller(error) => error.fmt(f),
            Self::Ui(error) => error.fmt(f),
            Self::Busy => f.write_str("document I/O already pending"),
            Self::UnsavedChanges => f.write_str("save or discard document changes before loading"),
            Self::ReconciliationRequired => f.write_str("retry identical save or reload first"),
            Self::NoRetry => f.write_str("no failed save retained for retry"),
            Self::RevisionExhausted => f.write_str("document revision exhausted"),
            Self::StaleRevision { saved, current } => {
                write!(
                    f,
                    "document revision conflict: captured {saved}, current {current}"
                )
            }
            Self::SnapshotChanged => f.write_str("controller snapshot changed since submission"),
            Self::SupersededReceipt => f.write_str("document receipt belongs to another operation"),
            Self::Submit(error) => write!(f, "document I/O submission: {error:?}"),
        }
    }
}

impl core::error::Error for DocumentError {}

impl From<ControllerError> for DocumentError {
    fn from(value: ControllerError) -> Self {
        Self::Controller(value)
    }
}

impl From<DesktopUiError> for DocumentError {
    fn from(value: DesktopUiError) -> Self {
        Self::Ui(value)
    }
}

type RegistryFactory = dyn Fn() -> TypeRegistry + Send + Sync;

struct SchemaEntry {
    version: u32,
    factory: Box<RegistryFactory>,
}

/// Caller-owned allowlist. One current, explicitly versioned schema per type
/// path. Factories must restore all nested types and editor validation metadata.
#[derive(Default)]
pub struct DocumentSchemaCatalog {
    entries: BTreeMap<String, SchemaEntry>,
}

impl DocumentSchemaCatalog {
    /// Creates an empty schema allowlist; no business types are built in.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registration never silently replaces a schema. The factory is trusted
    /// application code, not data loaded from the document.
    pub fn register<F>(
        &mut self,
        type_path: impl Into<String>,
        version: u32,
        factory: F,
    ) -> Result<(), DocumentError>
    where
        F: Fn() -> TypeRegistry + Send + Sync + 'static,
    {
        let path = type_path.into();
        if self.entries.contains_key(&path) {
            return Err(DocumentError::DuplicateSchema(path));
        }
        if path.is_empty() || factory().get_with_name(&path).is_none() {
            return Err(DocumentError::Registry(path));
        }
        self.entries.insert(
            path,
            SchemaEntry {
                version,
                factory: Box::new(factory),
            },
        );
        Ok(())
    }

    fn lookup(&self, path: &str, version: u32) -> Result<&SchemaEntry, DocumentError> {
        self.entries
            .get(path)
            .filter(|entry| entry.version == version)
            .ok_or_else(|| DocumentError::UnknownSchema {
                type_path: path.into(),
                version,
            })
    }
}

/// An entirely validated controller, not yet published to a host or UI.
pub struct DecodedDocument {
    controller: EditorController,
    source_model_revision: u64,
}

impl DecodedDocument {
    /// Borrows the fully validated, clean candidate controller.
    pub fn controller(&self) -> &EditorController {
        &self.controller
    }

    /// Historical source revision for diagnostics, not the restored session's
    /// model revision and not the provider's durable revision.
    pub const fn source_model_revision(&self) -> u64 {
        self.source_model_revision
    }

    /// Takes the validated candidate without altering any existing host state.
    pub fn into_controller(self) -> EditorController {
        self.controller
    }
}

/// Stable, engine-neutral codec; no paths, callbacks or Rust `TypeId`s on disk.
pub struct DocumentCodec;

impl DocumentCodec {
    /// Captures every registered object and complete `Outliner` hierarchy using
    /// the caller's schema versions, rejecting resource excess before storage.
    pub fn encode(
        controller: &EditorController,
        catalog: &DocumentSchemaCatalog,
        limits: DocumentLimits,
    ) -> Result<Vec<u8>, DocumentError> {
        if controller.has_active_edit() {
            return Err(DocumentError::Busy);
        }
        limits.validate()?;
        let objects = controller.document_objects();
        bound(objects.len(), limits.max_objects, "objects")?;
        let rows = controller.document_outliner_nodes();
        bound(rows.len(), limits.max_outliner_nodes, "Outliner rows")?;
        let mut writer = Writer {
            bytes: Vec::new(),
            limits,
        };
        writer.put(MAGIC)?;
        writer.put(&VERSION.to_le_bytes())?;
        writer.put(&controller.model_revision().to_le_bytes())?;
        writer.count(objects.len())?;
        writer.count(rows.len())?;
        let mut total_nodes = 0;
        for (item, root, registry) in objects {
            let path = root.type_name();
            let schema = catalog
                .entries
                .get(path)
                .ok_or_else(|| DocumentError::UnknownSchema {
                    type_path: path.into(),
                    version: 0,
                })?;
            if registry.get_with_name(path).is_none() {
                return Err(DocumentError::Registry(path.into()));
            }
            let mut estimated = 0;
            reflect_budget(root, 1, &mut total_nodes, &mut estimated, limits)?;
            let binary = to_binary(root).map_err(|e| DocumentError::Reflection(e.to_string()))?;
            bound(binary.len(), limits.max_object_bytes, "object binary bytes")?;
            writer.identity(item)?;
            writer.text(path)?;
            writer.put(&schema.version.to_le_bytes())?;
            writer.blob(&binary)?;
        }
        for row in rows {
            writer.put(&row.id().raw().to_le_bytes())?;
            writer.put(&[u8::from(row.parent().is_some())])?;
            if let Some(parent) = row.parent() {
                writer.put(&parent.raw().to_le_bytes())?;
            }
            writer.text(row.label())?;
            writer.put(&[u8::from(row.selection().is_some())])?;
            if let Some(item) = row.selection() {
                writer.identity(item)?;
            }
            writer.put(&[u8::from(row.is_expanded()), u8::from(row.is_visible())])?;
        }
        Ok(writer.bytes)
    }

    /// Constructs and validates every object/row off to the side. In particular,
    /// a valid early object cannot partially replace an existing controller if
    /// a later binary/schema/row fails validation.
    pub fn decode(
        bytes: &[u8],
        catalog: &DocumentSchemaCatalog,
        limits: DocumentLimits,
        controller_limits: ControllerLimits,
    ) -> Result<DecodedDocument, DocumentError> {
        limits.validate()?;
        bound(bytes.len(), limits.max_document_bytes, "document bytes")?;
        let mut reader = Reader { bytes, limits };
        if reader.take(8)? != MAGIC || reader.u32()? != VERSION {
            return Err(DocumentError::BadFormat("magic/version"));
        }
        let source_model_revision = reader.u64()?;
        let object_count = reader.u32()? as usize;
        let row_count = reader.u32()? as usize;
        bound(
            object_count,
            limits.max_objects.min(controller_limits.max_objects),
            "objects",
        )?;
        bound(
            row_count,
            limits
                .max_outliner_nodes
                .min(controller_limits.max_outliner_nodes),
            "Outliner rows",
        )?;
        let mut candidate = EditorController::with_limits(controller_limits);
        let mut objects = BTreeSet::new();
        let mut rows = BTreeSet::new();
        let mut total_nodes = 0;
        for _ in 0..object_count {
            let item = reader.identity()?;
            if !objects.insert(item.clone()) {
                return Err(DocumentError::DuplicateObject(item));
            }
            let path = reader.text()?;
            let version = reader.u32()?;
            let schema = catalog.lookup(path, version)?;
            let binary = reader.blob(limits.max_object_bytes, "object binary bytes")?;
            let registry = (schema.factory)();
            let target = registry
                .get_with_name(path)
                .ok_or_else(|| DocumentError::Registry(path.into()))?
                .type_info();
            let reflect_limits =
                DeserializeLimits::new(limits.max_reflect_depth, limits.max_collection_len);
            let root = from_binary_with_limits(binary, &registry, target, &reflect_limits)
                .map_err(|e| DocumentError::Reflection(e.to_string()))?;
            let mut estimated = 0;
            reflect_budget(&*root, 1, &mut total_nodes, &mut estimated, limits)?;
            candidate.register_object(item, root, registry)?;
        }
        for _ in 0..row_count {
            let id = OutlinerId::from_raw(reader.u128()?);
            if !rows.insert(id) {
                return Err(DocumentError::DuplicateOutliner(id));
            }
            let parent = if reader.boolean()? {
                Some(OutlinerId::from_raw(reader.u128()?))
            } else {
                None
            };
            // Parent-before-child format rejects missing parents and cycles.
            if parent.is_some_and(|parent| parent == id || !rows.contains(&parent)) {
                return Err(DocumentError::BadFormat("Outliner parent order/cycle"));
            }
            let mut row = OutlinerNode::new(id, parent, reader.text()?);
            if reader.boolean()? {
                let item = reader.identity()?;
                if !objects.contains(&item) {
                    return Err(DocumentError::BadFormat("Outliner object missing"));
                }
                row.set_selection(item);
            }
            row.set_expanded(reader.boolean()?);
            row.set_visible(reader.boolean()?);
            candidate.register_outliner_node(row)?;
        }
        if !reader.bytes.is_empty() {
            return Err(DocumentError::BadFormat("trailing bytes"));
        }
        Ok(DecodedDocument {
            controller: candidate,
            source_model_revision,
        })
    }
}

fn bound(actual: usize, limit: usize, name: &'static str) -> Result<(), DocumentError> {
    if actual > limit {
        Err(DocumentError::Limit(name))
    } else {
        Ok(())
    }
}

// Preflight recursive serialization before allocating its buffer. The estimate
// charges 32 bytes per node plus string bytes (an intentionally conservative
// bound for Prism binary node/primitive/count tags). Total reflected nodes are
// shared across all roots; per-object serialization has a separate byte limit.
fn reflect_budget(
    value: &dyn Reflect,
    depth: usize,
    total: &mut usize,
    estimated: &mut usize,
    limits: DocumentLimits,
) -> Result<(), DocumentError> {
    bound(depth, limits.max_reflect_depth, "reflection depth")?;
    *total = total
        .checked_add(1)
        .ok_or(DocumentError::Limit("reflection nodes"))?;
    bound(*total, limits.max_reflect_nodes, "reflection nodes")?;
    let text_bytes = value.downcast_ref::<String>().map_or(0, String::len);
    bound(text_bytes, limits.max_string_bytes, "model string bytes")?;
    *estimated = estimated
        .checked_add(32)
        .and_then(|v| v.checked_add(text_bytes))
        .ok_or(DocumentError::Limit("object allocation estimate"))?;
    bound(
        *estimated,
        limits.max_object_bytes,
        "object allocation estimate",
    )?;
    let mut child =
        |value: &dyn Reflect| reflect_budget(value, depth + 1, total, estimated, limits);
    match value.reflect_ref() {
        ReflectRef::Struct(v) => {
            for i in 0..v.field_count() {
                if let Some(v) = v.field_at(i) {
                    child(v)?;
                }
            }
        }
        ReflectRef::TupleStruct(v) => {
            for i in 0..v.field_count() {
                if let Some(v) = v.field(i) {
                    child(v)?;
                }
            }
        }
        ReflectRef::Enum(v) => {
            for i in 0..v.field_count() {
                if let Some(v) = v.field_at(i) {
                    child(v)?;
                }
            }
        }
        ReflectRef::List(v) => {
            bound(v.len(), limits.max_collection_len, "collection length")?;
            for v in v.iter_reflect() {
                child(v)?;
            }
        }
        ReflectRef::Array(v) => {
            bound(v.len(), limits.max_collection_len, "collection length")?;
            for v in v.iter_reflect() {
                child(v)?;
            }
        }
        ReflectRef::Set(v) => {
            bound(v.len(), limits.max_collection_len, "collection length")?;
            for v in v.iter_reflect() {
                child(v)?;
            }
        }
        ReflectRef::Map(v) => {
            bound(v.len(), limits.max_collection_len, "collection length")?;
            for (key, value) in v.iter_reflect() {
                child(key)?;
                child(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

struct Writer {
    bytes: Vec<u8>,
    limits: DocumentLimits,
}

impl Writer {
    fn put(&mut self, bytes: &[u8]) -> Result<(), DocumentError> {
        let size = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or(DocumentError::Limit("document bytes"))?;
        bound(size, self.limits.max_document_bytes, "document bytes")?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn count(&mut self, count: usize) -> Result<(), DocumentError> {
        self.put(
            &u32::try_from(count)
                .map_err(|_| DocumentError::Limit("u32 length"))?
                .to_le_bytes(),
        )
    }
    fn blob(&mut self, bytes: &[u8]) -> Result<(), DocumentError> {
        self.count(bytes.len())?;
        self.put(bytes)
    }
    fn text(&mut self, text: &str) -> Result<(), DocumentError> {
        bound(text.len(), self.limits.max_string_bytes, "string bytes")?;
        self.blob(text.as_bytes())
    }
    fn identity(&mut self, item: &SelectionItem) -> Result<(), DocumentError> {
        match item.kind() {
            SelectionKind::Entity(id) => {
                self.put(&[0])?;
                self.put(&id.to_le_bytes())
            }
            SelectionKind::Asset(id) => {
                self.put(&[1])?;
                self.put(&id.to_le_bytes())
            }
            SelectionKind::SubObject {
                owner,
                domain,
                index,
            } => {
                self.put(&[2])?;
                self.put(&owner.to_le_bytes())?;
                self.text(domain)?;
                self.put(&index.to_le_bytes())
            }
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    limits: DocumentLimits,
}

impl<'a> Reader<'a> {
    fn take(&mut self, size: usize) -> Result<&'a [u8], DocumentError> {
        if size > self.bytes.len() {
            return Err(DocumentError::BadFormat("truncated"));
        }
        let (value, rest) = self.bytes.split_at(size);
        self.bytes = rest;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, DocumentError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| DocumentError::BadFormat("u32"))?,
        ))
    }
    fn u64(&mut self) -> Result<u64, DocumentError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| DocumentError::BadFormat("u64"))?,
        ))
    }
    fn u128(&mut self) -> Result<u128, DocumentError> {
        Ok(u128::from_le_bytes(
            self.take(16)?
                .try_into()
                .map_err(|_| DocumentError::BadFormat("u128"))?,
        ))
    }
    fn boolean(&mut self) -> Result<bool, DocumentError> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(DocumentError::BadFormat("boolean")),
        }
    }
    fn blob(&mut self, limit: usize, name: &'static str) -> Result<&'a [u8], DocumentError> {
        let len = self.u32()? as usize;
        bound(len, limit, name)?;
        self.take(len)
    }
    fn text(&mut self) -> Result<&'a str, DocumentError> {
        let bytes = self.blob(self.limits.max_string_bytes, "string bytes")?;
        core::str::from_utf8(bytes).map_err(|_| DocumentError::BadFormat("UTF-8"))
    }
    fn identity(&mut self) -> Result<SelectionItem, DocumentError> {
        match self.take(1)?[0] {
            0 => Ok(SelectionItem::entity(self.u128()?)),
            1 => Ok(SelectionItem::asset(self.u128()?)),
            2 => {
                let owner = self.u128()?;
                let domain = self.text()?.to_owned();
                Ok(SelectionItem::sub_object(owner, domain, self.u32()?))
            }
            _ => Err(DocumentError::BadFormat("SelectionKind tag")),
        }
    }
}

#[derive(Clone)]
struct SnapshotGuard {
    model_revision: u64,
    bytes: Arc<[u8]>,
}

impl SnapshotGuard {
    fn capture(
        controller: &EditorController,
        catalog: &DocumentSchemaCatalog,
        limits: DocumentLimits,
    ) -> Result<Self, DocumentError> {
        Ok(Self {
            model_revision: controller.model_revision(),
            bytes: DocumentCodec::encode(controller, catalog, limits)?.into(),
        })
    }
    fn check(
        &self,
        controller: &EditorController,
        catalog: &DocumentSchemaCatalog,
        limits: DocumentLimits,
    ) -> Result<(), DocumentError> {
        if controller.model_revision() != self.model_revision {
            return Err(DocumentError::StaleRevision {
                saved: self.model_revision,
                current: controller.model_revision(),
            });
        }
        // Also fences registration/replacement APIs that do not bump model_revision.
        if DocumentCodec::encode(controller, catalog, limits)?.as_slice() != &*self.bytes {
            return Err(DocumentError::SnapshotChanged);
        }
        Ok(())
    }
}

#[derive(Clone)]
struct PendingSave {
    request: CellSaveRequest,
    guard: SnapshotGuard,
}

enum Pending {
    Save(PendingSave),
    Load(SnapshotGuard),
}

/// Completion returned without silently clearing a controller's dirty state.
pub struct DocumentSaveReceipt {
    /// Identity of the accepted save request.
    pub request_id: IoRequestId,
    /// Provider revision actually committed, independent of model revision.
    pub durable_revision: u64,
    guard: SnapshotGuard,
    limits: DocumentLimits,
}

impl DocumentSaveReceipt {
    /// Model revision captured with the saved bytes.
    pub const fn model_revision(&self) -> u64 {
        self.guard.model_revision
    }

    /// Marks clean only if both the model revision and full snapshot still match.
    pub fn acknowledge(
        &self,
        controller: &mut EditorController,
        catalog: &DocumentSchemaCatalog,
    ) -> Result<(), DocumentError> {
        self.guard.check(controller, catalog, self.limits)?;
        controller.mark_saved(self.guard.model_revision)?;
        Ok(())
    }

    /// Confirms a save without acquiring mutable controller access or cancelling
    /// an in-progress text/IME draft. Failed guards leave the UI untouched.
    pub fn acknowledge_ui(
        &self,
        ui: &mut EditorDesktopUi,
        catalog: &DocumentSchemaCatalog,
    ) -> Result<(), DocumentError> {
        self.guard.check(ui.controller(), catalog, self.limits)?;
        ui.mark_saved(self.guard.model_revision)?;
        Ok(())
    }
}

/// Loaded bytes awaiting complete schema/controller validation. No live state
/// has changed. Dropping this receipt leaves the old durable base unchanged.
pub struct DocumentLoadReceipt {
    /// Identity of the accepted load request.
    pub request_id: IoRequestId,
    durable_revision: u64,
    bytes: Vec<u8>,
    guard: SnapshotGuard,
    owner: Arc<()>,
    generation: u64,
    limits: DocumentLimits,
}

impl DocumentLoadReceipt {
    /// Decodes and validates all objects, schemas and rows without modifying
    /// the live controller, UI or adapter's durable base. This is bounded CPU work.
    pub fn prepare(
        self,
        catalog: &DocumentSchemaCatalog,
        controller_limits: ControllerLimits,
    ) -> Result<PreparedDocumentLoad, DocumentError> {
        let decoded = DocumentCodec::decode(&self.bytes, catalog, self.limits, controller_limits)?;
        Ok(PreparedDocumentLoad {
            decoded,
            receipt: self,
        })
    }
}

/// Validated candidate. Hosts may prepare their replacement UI before install.
pub struct PreparedDocumentLoad {
    decoded: DecodedDocument,
    receipt: DocumentLoadReceipt,
}

impl PreparedDocumentLoad {
    /// Borrows the fully validated controller for candidate UI preparation.
    pub fn controller(&self) -> &EditorController {
        self.decoded.controller()
    }
    /// Historical model revision stored in the source document.
    pub const fn source_model_revision(&self) -> u64 {
        self.decoded.source_model_revision
    }
    /// Durable provider revision read with this document.
    pub const fn durable_revision(&self) -> u64 {
        self.receipt.durable_revision
    }

    fn validate_install(
        &self,
        adapter: &DocumentFileAdapter,
        controller: &EditorController,
        catalog: &DocumentSchemaCatalog,
    ) -> Result<(), DocumentError> {
        if !Arc::ptr_eq(&self.receipt.owner, &adapter.owner)
            || self.receipt.generation != adapter.generation
            || adapter.is_busy()
        {
            return Err(DocumentError::SupersededReceipt);
        }
        self.receipt
            .guard
            .check(controller, catalog, self.receipt.limits)
    }

    /// Publishes the candidate and durable base only after validating the owner,
    /// operation generation and unchanged live snapshot. Host command factories
    /// are retained after budget validation; callbacks and history are never
    /// read from the document. Model history resets on installation.
    pub fn install(
        self,
        adapter: &mut DocumentFileAdapter,
        controller: &mut EditorController,
        catalog: &DocumentSchemaCatalog,
    ) -> Result<(), DocumentError> {
        self.validate_install(adapter, controller, catalog)?;
        let mut candidate = self.decoded.into_controller();
        candidate.inherit_commands(controller)?;
        // All fallible work precedes either mutation. These two assignments are
        // the host's publication point; old controller is dropped only here.
        *controller = candidate;
        adapter.durable_revision = Some(self.receipt.durable_revision);
        adapter.retry = None;
        adapter.reconciliation_required = false;
        Ok(())
    }

    /// Validates the receipt, constructs and lays out an entirely new retained
    /// UI at the old logical size, then publishes UI and adapter state together.
    /// Every recoverable failure leaves the original UI, drafts and durable base
    /// unchanged. A successful load intentionally resets transient UI/history.
    pub fn install_ui(
        self,
        adapter: &mut DocumentFileAdapter,
        ui: &mut EditorDesktopUi,
        catalog: &DocumentSchemaCatalog,
    ) -> Result<(), DocumentError> {
        self.validate_install(adapter, ui.controller(), catalog)?;
        let size = ui.size();
        let mut controller = self.decoded.into_controller();
        controller.inherit_commands(ui.controller())?;
        let mut candidate = EditorDesktopUi::new(controller)?;
        candidate.resize(size.width, size.height)?;
        // No fallible work or user callbacks after the publication point.
        // Keep the old UI alive until both adapter and UI state are published.
        let old = std::mem::replace(ui, candidate);
        adapter.durable_revision = Some(self.receipt.durable_revision);
        adapter.retry = None;
        adapter.reconciliation_required = false;
        drop(old);
        Ok(())
    }
}

/// One terminal event per accepted request; no failed operation marks clean.
pub enum DocumentIoEvent {
    /// Bytes were committed; acknowledgement is an explicit separate step.
    Saved(DocumentSaveReceipt),
    /// Bytes were read but have not yet passed document/schema validation.
    Loaded(DocumentLoadReceipt),
    /// An accepted operation failed; live controller/UI state was not mutated.
    Failed {
        /// Identity of the failed accepted request.
        request_id: IoRequestId,
        /// Typed driver/provider failure, including conflicts and uncertain commits.
        error: IoFailure<FileCellIoError>,
    },
    /// Cancellation completed; a save still requires reconciliation.
    Cancelled {
        /// Identity of the cancelled accepted request.
        request_id: IoRequestId,
    },
}

/// Single-document async adapter. Identity and load token come from the world
/// owner; callers cannot inject an arbitrary filename or traverse the root.
pub struct DocumentFileAdapter {
    driver: WorldIoDriver<FileCellIoProvider>,
    load_request: CellLoadRequest,
    limits: DocumentLimits,
    pending: Option<(IoRequestId, Pending)>,
    retry: Option<PendingSave>,
    durable_revision: Option<u64>,
    reconciliation_required: bool,
    owner: Arc<()>,
    generation: u64,
}

impl DocumentFileAdapter {
    /// Binds one world-issued cell identity to an asynchronous production
    /// provider. The provider should be constructed outside the frame loop.
    /// Its payload limit and request read budget must cover the document limit.
    pub fn new(
        provider: FileCellIoProvider,
        load_request: CellLoadRequest,
        limits: DocumentLimits,
    ) -> Result<Self, DocumentError> {
        limits.validate()?;
        if load_request.max_bytes() < limits.max_document_bytes
            || provider.limits().max_payload_bytes() < limits.max_document_bytes
        {
            return Err(DocumentError::InvalidLimits);
        }
        let budget =
            IoBudget::new(1, load_request.max_bytes()).map_err(|_| DocumentError::InvalidLimits)?;
        Ok(Self {
            driver: WorldIoDriver::new(provider, budget),
            load_request,
            limits,
            pending: None,
            retry: None,
            durable_revision: None,
            reconciliation_required: false,
            owner: Arc::new(()),
            generation: 0,
        })
    }

    /// Whether an accepted request still awaits its terminal poll event.
    pub fn is_busy(&self) -> bool {
        self.pending.is_some()
    }
    /// Last successful save or installed load; a failed load never resets it.
    pub const fn durable_revision(&self) -> Option<u64> {
        self.durable_revision
    }
    /// Whether only identical retry or explicit reload/install may precede a new save.
    pub const fn reconciliation_required(&self) -> bool {
        self.reconciliation_required
    }

    fn next_generation(&self) -> Result<u64, DocumentError> {
        if self.is_busy() {
            return Err(DocumentError::Busy);
        }
        self.generation
            .checked_add(1)
            .ok_or(DocumentError::RevisionExhausted)
    }

    /// Captures a bounded snapshot and submits filesystem work to a worker;
    /// does not wait for I/O or clear dirty state. Fresh adapters cannot overwrite
    /// different existing documents; load/install first to establish their base.
    pub fn submit_save(
        &mut self,
        controller: &EditorController,
        catalog: &DocumentSchemaCatalog,
    ) -> Result<IoRequestId, DocumentError> {
        let generation = self.next_generation()?;
        if self.reconciliation_required {
            return Err(DocumentError::ReconciliationRequired);
        }
        let revision = self
            .durable_revision
            .map_or(Some(0), |base| base.checked_add(1))
            .ok_or(DocumentError::RevisionExhausted)?;
        let guard = SnapshotGuard::capture(controller, catalog, self.limits)?;
        let mut request = CellSaveRequest::new(
            self.load_request.cell(),
            self.load_request.file().clone(),
            revision,
        );
        if let Some(base) = self.durable_revision {
            request = request.with_base_revision(base);
        }
        let id = self
            .driver
            .submit_save(request.clone(), guard.bytes.to_vec())
            .map_err(DocumentError::Submit)?;
        self.generation = generation;
        self.pending = Some((id, Pending::Save(PendingSave { request, guard })));
        Ok(id)
    }

    /// Retries exactly the retained bytes/revision/base, including uncertain
    /// post-rename failures. It cannot acknowledge subsequent model edits.
    pub fn retry_save(&mut self) -> Result<IoRequestId, DocumentError> {
        let generation = self.next_generation()?;
        let saved = self.retry.as_ref().ok_or(DocumentError::NoRetry)?.clone();
        let id = self
            .driver
            .submit_save(saved.request.clone(), saved.guard.bytes.to_vec())
            .map_err(DocumentError::Submit)?;
        self.generation = generation;
        self.pending = Some((id, Pending::Save(saved)));
        Ok(id)
    }

    /// Captures the live-state guard and submits a bounded filesystem read.
    /// The terminal receipt still needs prepare and install before publication.
    pub fn submit_load(
        &mut self,
        controller: &EditorController,
        catalog: &DocumentSchemaCatalog,
    ) -> Result<IoRequestId, DocumentError> {
        let generation = self.next_generation()?;
        let guard = SnapshotGuard::capture(controller, catalog, self.limits)?;
        let id = self
            .driver
            .submit_load(self.load_request.clone())
            .map_err(DocumentError::Submit)?;
        self.generation = generation;
        self.pending = Some((id, Pending::Load(guard)));
        Ok(id)
    }

    /// Cancellation queues a terminal event. Poll it before submitting again.
    /// Like the underlying driver, cancellation may wait for a commit that has
    /// already entered its publication fence, and cannot roll that commit back.
    pub fn cancel(&mut self, request_id: IoRequestId) -> bool {
        self.driver.cancel(request_id)
    }

    /// Polls without filesystem access or decoding, registering the supplied
    /// waker while pending. Returns None only when no request/event remains.
    pub fn poll(&mut self, context: &mut Context<'_>) -> Poll<Option<DocumentIoEvent>> {
        let completion = match self.driver.poll(context) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(None) => return Poll::Ready(None),
            Poll::Ready(Some(completion)) => completion,
        };
        let (id, pending) = self
            .pending
            .take()
            .expect("private driver completion has a pending document operation");
        assert_eq!(
            id,
            completion.request_id(),
            "private driver completion matches document request"
        );
        let event = match completion {
            IoCompletion::Saved {
                request_id,
                request,
            } => {
                let Pending::Save(saved) = pending else {
                    unreachable!("private driver save completes a pending save");
                };
                self.durable_revision = Some(request.revision());
                self.retry = None;
                self.reconciliation_required = false;
                DocumentIoEvent::Saved(DocumentSaveReceipt {
                    request_id,
                    durable_revision: request.revision(),
                    guard: saved.guard,
                    limits: self.limits,
                })
            }
            IoCompletion::Loaded {
                request_id,
                bytes,
                revision,
                ..
            } => {
                let Pending::Load(guard) = pending else {
                    unreachable!("private driver load completes a pending load");
                };
                let Some(durable_revision) = revision else {
                    return Poll::Ready(Some(DocumentIoEvent::Failed {
                        request_id,
                        error: IoFailure::Provider(FileCellIoError::InvalidRequest),
                    }));
                };
                DocumentIoEvent::Loaded(DocumentLoadReceipt {
                    request_id,
                    durable_revision,
                    bytes,
                    guard,
                    owner: self.owner.clone(),
                    generation: self.generation,
                    limits: self.limits,
                })
            }
            IoCompletion::Failed {
                request_id, error, ..
            } => {
                if let Pending::Save(saved) = pending {
                    self.retry = Some(saved);
                    self.reconciliation_required = true;
                }
                DocumentIoEvent::Failed { request_id, error }
            }
            IoCompletion::Cancelled { request_id, .. } => {
                if let Pending::Save(saved) = pending {
                    self.retry = Some(saved);
                    self.reconciliation_required = true;
                }
                DocumentIoEvent::Cancelled { request_id }
            }
        };
        Poll::Ready(Some(event))
    }
}
