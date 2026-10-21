//! Bounded read-only presentation of the core editable semantic diff.

use crate::{args, error::CommandError};
use prism_asset_build_core::{
    diff_editable_assets, ContentHash, EditableAsset, EditableDependency, EditableDependencyKind,
    EditableDiffImpact, EditablePayload, EditableSubAsset, EditableSubAssetDiff,
    EditableSubAssetReference, EditableValueChange, SourceRecord,
};
use serde::Serialize;
use std::fs::{self, File, Metadata};
use std::io::{self, Read, Write};
use std::path::Path;

const MAX_INPUT_BYTES: u64 = 8 * 1024 * 1024;
// These are presentation limits in addition to the core's per-input limits.
// Reject over-budget results rather than truncate or claim an incomplete diff.
const MAX_CHANGE_RECORDS: usize = 65_536;
const MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
// Leave room for both JSONL v1 event envelopes and their terminating newlines.
const MAX_RESULT_BYTES: usize = MAX_OUTPUT_BYTES - 1024;

fn invalid(message: impl Into<String>) -> CommandError {
    CommandError::InvalidInput(message.into())
}

#[derive(Debug, Serialize)]
struct SnapshotResult {
    asset_id: String,
    bytes: u64,
    content_hash: String,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Impact {
    cook: bool,
    editor_import: bool,
}

impl From<EditableDiffImpact> for Impact {
    fn from(value: EditableDiffImpact) -> Self {
        Self {
            cook: value.cook,
            editor_import: value.editor_import,
        }
    }
}

const COOK: Impact = Impact {
    cook: true,
    editor_import: false,
};
const EDITOR: Impact = Impact {
    cook: false,
    editor_import: true,
};

#[derive(Debug, Serialize)]
struct Change<T> {
    before: T,
    after: T,
    impact: Impact,
}

fn project_change<T, U>(
    change: Option<EditableValueChange<T>>,
    project: impl Fn(T) -> U,
    impact: Impact,
) -> Option<Change<U>> {
    change.map(|value| Change {
        before: project(value.before),
        after: project(value.after),
        impact,
    })
}

#[derive(Debug, Serialize)]
struct BlobResult {
    bytes: usize,
    content_hash: String,
}

impl BlobResult {
    fn new(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.len(),
            content_hash: ContentHash::digest(bytes).to_hex(),
        }
    }
}

#[derive(Debug, Serialize)]
struct PayloadResult {
    bytes: u64,
    content_hash: String,
    storage: &'static str,
}

impl From<&EditablePayload> for PayloadResult {
    fn from(payload: &EditablePayload) -> Self {
        Self {
            bytes: payload.byte_len(),
            content_hash: payload.content_hash().to_hex(),
            storage: match payload {
                EditablePayload::Inline(_) => "inline",
                EditablePayload::External(_) => "external",
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct SourceResult {
    uri: String,
    format: String,
    content_hash: String,
    embedded_source: Option<BlobResult>,
}

impl From<&SourceRecord> for SourceResult {
    fn from(source: &SourceRecord) -> Self {
        Self {
            uri: source.uri.clone(),
            format: source.format.clone(),
            content_hash: source.content_hash.to_hex(),
            embedded_source: source.embedded_source.as_deref().map(BlobResult::new),
        }
    }
}

#[derive(Debug, Serialize)]
struct SubAssetResult {
    name: String,
    kind: String,
}

impl From<&EditableSubAsset> for SubAssetResult {
    fn from(value: &EditableSubAsset) -> Self {
        Self {
            name: value.name.clone(),
            kind: value.kind.to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
struct SubAssetChange {
    id: String,
    change: &'static str,
    before: Option<SubAssetResult>,
    after: Option<SubAssetResult>,
    impact: Impact,
}

impl From<&EditableSubAssetDiff<'_>> for SubAssetChange {
    fn from(value: &EditableSubAssetDiff<'_>) -> Self {
        let (id, change, before, after) = match value {
            EditableSubAssetDiff::Added(after) => (after.id, "added", None, Some(*after)),
            EditableSubAssetDiff::Removed(before) => (before.id, "removed", Some(*before), None),
            EditableSubAssetDiff::Modified { before, after } => {
                (before.id, "modified", Some(*before), Some(*after))
            }
        };
        Self {
            id: id.to_string(),
            change,
            before: before.map(SubAssetResult::from),
            after: after.map(SubAssetResult::from),
            impact: COOK,
        }
    }
}

fn edge_kind(kind: EditableDependencyKind) -> (&'static str, Impact) {
    match kind {
        EditableDependencyKind::Hard => ("hard", COOK),
        EditableDependencyKind::Soft => ("soft", COOK),
        EditableDependencyKind::EditorOnly => ("editor_only", EDITOR),
    }
}

#[derive(Debug, Serialize)]
struct DependencyResult {
    asset_id: String,
    kind: &'static str,
    impact: Impact,
}

impl From<&EditableDependency> for DependencyResult {
    fn from(edge: &EditableDependency) -> Self {
        let (kind, impact) = edge_kind(edge.kind);
        Self {
            asset_id: edge.asset.0.to_string(),
            kind,
            impact,
        }
    }
}

#[derive(Debug, Serialize)]
struct ReferenceResult {
    asset_id: String,
    subasset_id: String,
    kind: &'static str,
    impact: Impact,
}

impl From<&EditableSubAssetReference> for ReferenceResult {
    fn from(edge: &EditableSubAssetReference) -> Self {
        let (kind, impact) = edge_kind(edge.kind);
        Self {
            asset_id: edge.asset.0.to_string(),
            subasset_id: edge.subasset.to_string(),
            kind,
            impact,
        }
    }
}

#[derive(Debug, Serialize)]
struct SetChanges<T> {
    added: Vec<T>,
    removed: Vec<T>,
}

#[derive(Debug, Serialize)]
pub struct EditableDiffResult {
    before: SnapshotResult,
    after: SnapshotResult,
    identical: bool,
    /// Local saved inputs only; external artifacts/toolchains are not compared.
    impact: Impact,
    asset_id: Option<Change<String>>,
    kind: Option<Change<String>>,
    flags: Option<Change<u32>>,
    canonical_settings: Option<Change<BlobResult>>,
    source: Option<Change<SourceResult>>,
    authoring_payload: Option<Change<PayloadResult>>,
    preview: Option<Change<Option<BlobResult>>>,
    subassets: Vec<SubAssetChange>,
    dependencies: SetChanges<DependencyResult>,
    subasset_references: SetChanges<ReferenceResult>,
}

pub fn run(command: args::EditableDiff) -> Result<EditableDiffResult, CommandError> {
    let limit = command.max_input_bytes.unwrap_or(MAX_INPUT_BYTES);
    if limit == 0 || limit > MAX_INPUT_BYTES {
        return Err(invalid(
            "--max-input-bytes must be positive and at most 8388608",
        ));
    }
    let (before, before_summary) = read_snapshot(&command.before, limit, "before")?;
    let (after, after_summary) = read_snapshot(&command.after, limit, "after")?;
    // All matching, canonical validation and equality decisions belong to core.
    let diff = diff_editable_assets(&before, &after)?;
    let records = diff.subassets.len()
        + diff.dependencies.added.len()
        + diff.dependencies.removed.len()
        + diff.subasset_references.added.len()
        + diff.subasset_references.removed.len();
    if records > MAX_CHANGE_RECORDS {
        return Err(invalid(
            "editable diff exceeds 65536 collection change records",
        ));
    }
    let result = EditableDiffResult {
        before: before_summary,
        after: after_summary,
        identical: diff.is_empty(),
        impact: diff.impact().into(),
        asset_id: project_change(diff.asset, |id| id.0.to_string(), COOK),
        kind: project_change(diff.kind, |kind| kind.to_string(), COOK),
        flags: project_change(diff.flags, |flags| flags, COOK),
        canonical_settings: project_change(diff.canonical_settings, BlobResult::new, COOK),
        source: project_change(diff.source, SourceResult::from, EDITOR),
        authoring_payload: project_change(
            diff.authoring_payload,
            PayloadResult::from,
            diff.authoring_payload_impact().into(),
        ),
        preview: project_change(diff.preview, |value| value.map(BlobResult::new), EDITOR),
        subassets: diff.subassets.iter().map(SubAssetChange::from).collect(),
        dependencies: SetChanges {
            added: diff
                .dependencies
                .added
                .iter()
                .map(|edge| DependencyResult::from(*edge))
                .collect(),
            removed: diff
                .dependencies
                .removed
                .iter()
                .map(|edge| DependencyResult::from(*edge))
                .collect(),
        },
        subasset_references: SetChanges {
            added: diff
                .subasset_references
                .added
                .iter()
                .map(|edge| ReferenceResult::from(*edge))
                .collect(),
            removed: diff
                .subasset_references
                .removed
                .iter()
                .map(|edge| ReferenceResult::from(*edge))
                .collect(),
        },
    };
    // Measure escaped serialization, not source string lengths. Check both
    // modes before handing the result to Output, so a failure has one terminal
    // failed event and never a partially written success event.
    serde_json::to_writer(&mut OutputBudget::default(), &result)
        .map_err(|_| invalid("editable diff exceeds the 8 MiB output budget"))?;
    result
        .write_human(&mut OutputBudget::default())
        .map_err(|_| invalid("editable diff exceeds the 8 MiB output budget"))?;
    Ok(result)
}

fn regular_metadata(path: &Path, side: &str) -> Result<Metadata, CommandError> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(invalid(format!(
            "{side} input must be a regular non-symlink .prism file"
        )));
    }
    Ok(metadata)
}

fn same_metadata(left: &Metadata, right: &Metadata) -> bool {
    if left.len() != right.len() || left.modified().ok() != right.modified().ok() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if left.dev() != right.dev() || left.ino() != right.ino() {
            return false;
        }
    }
    true
}

fn read_snapshot(
    path: &Path,
    limit: u64,
    side: &str,
) -> Result<(EditableAsset, SnapshotResult), CommandError> {
    if path
        .extension()
        .is_none_or(|extension| extension != "prism")
    {
        return Err(invalid(format!(
            "{side} input must have a .prism extension"
        )));
    }
    let initial = regular_metadata(path, side)?;
    if initial.len() > limit {
        return Err(invalid(format!("{side} input exceeds --max-input-bytes")));
    }
    let mut file = File::open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file() || opened.len() > limit {
        return Err(invalid(format!(
            "{side} opened input is not a regular file within --max-input-bytes"
        )));
    }
    if !same_metadata(&initial, &opened) {
        return Err(invalid(format!("{side} input changed while opening")));
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    (&mut file).take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(invalid(format!(
            "{side} input exceeds --max-input-bytes while reading"
        )));
    }
    let final_path = regular_metadata(path, side)?;
    if bytes.len() as u64 != opened.len()
        || !same_metadata(&opened, &file.metadata()?)
        || !same_metadata(&opened, &final_path)
    {
        return Err(invalid(format!("{side} input changed while reading")));
    }
    let asset = EditableAsset::decode(&bytes)?;
    let summary = SnapshotResult {
        asset_id: asset.asset.0.to_string(),
        bytes: bytes.len() as u64,
        content_hash: ContentHash::digest(&bytes).to_hex(),
    };
    Ok((asset, summary))
}

#[derive(Default)]
struct OutputBudget {
    bytes: usize,
}

impl Write for OutputBudget {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self
            .bytes
            .checked_add(bytes.len())
            .filter(|next| *next <= MAX_RESULT_BYTES)
            .ok_or_else(|| io::Error::other("editable diff output budget exceeded"))?;
        self.bytes = next;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn record(out: &mut impl Write, label: &str, value: &impl Serialize) -> io::Result<()> {
    write!(out, "{label}=")?;
    // JSON escaping also keeps user-controlled metadata on a single text line.
    serde_json::to_writer(&mut *out, value).map_err(io::Error::other)?;
    writeln!(out)
}

fn field<T: Serialize>(
    out: &mut impl Write,
    label: &str,
    value: &Option<Change<T>>,
) -> io::Result<()> {
    if let Some(value) = value {
        record(out, label, value)?;
    }
    Ok(())
}

impl EditableDiffResult {
    pub fn write_human(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(
            out,
            "identical={} cook={} editor_import={}",
            self.identical, self.impact.cook, self.impact.editor_import
        )?;
        record(out, "before", &self.before)?;
        record(out, "after", &self.after)?;
        field(out, "asset_id", &self.asset_id)?;
        field(out, "kind", &self.kind)?;
        field(out, "flags", &self.flags)?;
        field(out, "canonical_settings", &self.canonical_settings)?;
        field(out, "source", &self.source)?;
        field(out, "authoring_payload", &self.authoring_payload)?;
        field(out, "preview", &self.preview)?;
        for change in &self.subassets {
            record(out, "subasset", change)?;
        }
        for edge in &self.dependencies.added {
            record(out, "dependency_added", edge)?;
        }
        for edge in &self.dependencies.removed {
            record(out, "dependency_removed", edge)?;
        }
        for edge in &self.subasset_references.added {
            record(out, "reference_added", edge)?;
        }
        for edge in &self.subasset_references.removed {
            record(out, "reference_removed", edge)?;
        }
        Ok(())
    }
}
