//! Canonical, bounded `.prism` editable-asset packages.
//!
//! The package contract, source capture, reimport boundary, and local
//! workspace publication live here. Runtime artifacts remain outside this
//! editing boundary.

use crate::source::SourceSnapshot;
use crate::types::{AssetId, ContentHash};
use fs4::fs_std::FileExt;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC: &[u8] = b"PRISM-EDITABLE\0";
/// The canonical `.prism` editable-package schema.
pub const EDITABLE_ASSET_SCHEMA: u32 = 2;
/// Maximum accepted encoded package size.
pub const MAX_EDITABLE_ASSET_BYTES: usize = 64 * 1024 * 1024;
/// Hard aggregate encoded-package budget for workspace snapshots and reimport scans.
pub const MAX_EDITABLE_WORKSPACE_ENCODED_BYTES: u64 = 512 * 1024 * 1024;
const MAX_SETTINGS_BYTES: usize = 1024 * 1024;
const MAX_SOURCE_URI_BYTES: usize = 4096;
const MAX_SOURCE_FORMAT_BYTES: usize = 256;
const MAX_EMBEDDED_SOURCE_BYTES: usize = 32 * 1024 * 1024;
const MAX_EXTERNAL_SOURCE_BYTES: usize = 32 * 1024 * 1024;
const MAX_SUBASSETS: usize = 65_536;
const MAX_SUBASSET_NAME_BYTES: usize = 4096;
const MAX_DEPENDENCIES: usize = 65_536;
const MAX_AUTHORING_PAYLOAD_BYTES: usize = 32 * 1024 * 1024;
const MAX_PREVIEW_BYTES: usize = 8 * 1024 * 1024;
const MAX_SUBASSET_REFERENCES: usize = 65_536;
const MAX_WORKSPACE_PACKAGES: usize = 65_536;
// Counts include unrelated entries and directories (including the root). The
// reserved transaction subtree is recovered separately and is never imported.
const MAX_WORKSPACE_ENTRIES: usize = 262_144;
const MAX_WORKSPACE_DIRECTORIES: usize = 16_384;
const MAX_WORKSPACE_DIRECTORY_DEPTH: usize = 64;
const EDITABLE_TRANSACTION_DIRECTORY: &str = ".prism-editable-transactions";
const EDITABLE_TRANSACTION_MAGIC: &[u8] = b"PRISM-EDITABLE-TX\0";
const EDITABLE_TRANSACTION_SCHEMA: u32 = 1;
const EDITABLE_TRANSACTION_PREPARED: &[u8] = b"prepared";
const EDITABLE_TRANSACTION_COMMITTED: &[u8] = b"committed";
const MAX_EDITABLE_TRANSACTION_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_EDITABLE_TRANSACTION_MARKER_BYTES: usize = 32;
static EDITABLE_TRANSACTION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Stable directory entry for a source-package subasset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableSubAsset {
    pub id: u64,
    pub name: String,
    pub kind: u128,
}

/// An explicit redirect required when a referenced stable subasset is removed.
///
/// The importer owns stable ID derivation. This record never guesses that two
/// unrelated IDs represent a rename.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct EditableSubAssetRedirect {
    pub from: u64,
    pub to: u64,
}

/// One deterministic outcome while reconciling an imported subasset directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditableSubAssetChange {
    Retained {
        id: u64,
    },
    Renamed {
        id: u64,
        previous_name: String,
        next_name: String,
    },
    Added {
        subasset: EditableSubAsset,
    },
    Removed {
        subasset: EditableSubAsset,
    },
    Redirected {
        from: u64,
        to: u64,
    },
}

/// Caller-owned, side-effect-free result of a subasset reimport match.
///
/// `subassets` is the next canonical directory. Applying it, updating external
/// references, and publishing a new editable revision remain caller actions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableSubAssetReimportPlan {
    pub subassets: Vec<EditableSubAsset>,
    pub changes: Vec<EditableSubAssetChange>,
}

/// Semantic strength of an editable-asset dependency.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EditableDependencyKind {
    Hard,
    Soft,
    EditorOnly,
}

impl EditableDependencyKind {
    const fn tag(self) -> u8 {
        match self {
            Self::Hard => 0,
            Self::Soft => 1,
            Self::EditorOnly => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, EditableAssetError> {
        match tag {
            0 => Ok(Self::Hard),
            1 => Ok(Self::Soft),
            2 => Ok(Self::EditorOnly),
            _ => Err(EditableAssetError::InvalidDependencyKind),
        }
    }
}

/// One dependency declared by an editable asset.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct EditableDependency {
    pub asset: AssetId,
    pub kind: EditableDependencyKind,
}

/// A typed reference to one stable subasset owned by another editable asset.
///
/// This is deliberately separate from `authoring_payload`: generic workspace
/// publication can update these references without interpreting a
/// format-specific payload. Payload references are rewritten through the
/// caller-owned [`EditablePayloadReferenceRewriter`] contract.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct EditableSubAssetReference {
    pub asset: AssetId,
    pub subasset: u64,
    pub kind: EditableDependencyKind,
}

/// One explicit stable-subasset redirect applied during a workspace reimport.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct EditableSubAssetReferenceRewrite {
    pub asset: AssetId,
    pub from: u64,
    pub to: u64,
}

/// Caller-owned semantic rewrite for opaque authoring payloads.
///
/// The workspace invokes this only for packages whose declared typed
/// subasset references changed. Returning an error aborts the whole reimport
/// before any package is published.
pub trait EditablePayloadReferenceRewriter {
    /// Returns canonical replacement bytes for one affected package.
    fn rewrite_authoring_payload(
        &self,
        asset: &EditableAsset,
        rewrites: &[EditableSubAssetReferenceRewrite],
    ) -> Result<Vec<u8>, EditableAssetError>;
}

/// Caller-owned importer for a verified editable source snapshot.
pub trait EditableReimporter {
    /// Rebuilds editable fields while preserving the package identity, kind,
    /// flags, and source-record authority held by the workspace.
    fn reimport(
        &self,
        previous: &EditableAsset,
        source: &SourceSnapshot,
    ) -> Result<EditableReimportOutput, EditableAssetError>;
}

/// Reimporter output before workspace reconciliation and publication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableReimportOutput {
    pub canonical_settings: Vec<u8>,
    pub subassets: Vec<EditableSubAsset>,
    pub dependencies: Vec<EditableDependency>,
    pub subasset_references: Vec<EditableSubAssetReference>,
    pub authoring_payload: Vec<u8>,
    pub preview: Option<Vec<u8>>,
}

/// Result of a committed editable-workspace reimport.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableWorkspaceReimport {
    pub plan: EditableSubAssetReimportPlan,
    /// Every package whose encoded canonical bytes changed.
    pub updated_assets: Vec<AssetId>,
}

/// One replacement based on the canonical bytes captured by an editor.
///
/// The identity selects an existing workspace package; its path stays stable.
/// The expected hash is BLAKE3 of the entire previous encoded package, including
/// editor metadata, so even a preview-only concurrent edit cannot be overwritten.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableWorkspaceEdit {
    pub expected_hash: ContentHash,
    pub replacement: EditableAsset,
}

/// Receipt for a committed compare-and-save batch, ordered by stable identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableWorkspaceSave {
    /// Packages whose canonical bytes actually changed. No-op saves are omitted.
    pub updated_assets: Vec<AssetId>,
    /// Resulting hashes for every supplied edit, including no-ops. These become
    /// the expected hashes for subsequent saves or an explicit undo operation.
    pub content_hashes: BTreeMap<AssetId, ContentHash>,
}

/// Local root that owns a set of editable `.prism` packages.
///
/// The workspace is intentionally local and file-system scoped. It owns no
/// importer, semantic payload format, runtime artifact, or cache policy.
#[derive(Clone, Debug)]
pub struct EditableWorkspace {
    root: PathBuf,
}

/// Reimport information stored with the editable representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceRecord {
    /// A source-provider URI or canonical logical source path.
    pub uri: String,
    /// Stable importer-facing source format identifier.
    pub format: String,
    /// Digest of the exact embedded bytes, or of the externally captured bytes.
    pub content_hash: ContentHash,
    /// Optional self-contained source bytes.
    pub embedded_source: Option<Vec<u8>>,
}

/// Caller-owned boundary for resolving an external source record.
pub trait EditableSourceProvider {
    /// Captures an immutable source snapshot for the persisted source record.
    fn snapshot(&self, source: &SourceRecord) -> Result<SourceSnapshot, EditableAssetError>;
}

/// Bounded local-file source provider rooted at one content directory.
#[derive(Clone, Debug)]
pub struct LocalFileSourceProvider {
    root: PathBuf,
}

impl LocalFileSourceProvider {
    /// Opens a provider rooted at an existing, non-symlink directory.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, EditableAssetError> {
        let root = root.into();
        ensure_directory(&root)?;
        Ok(Self { root })
    }

    /// Returns the configured content root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl EditableSourceProvider for LocalFileSourceProvider {
    fn snapshot(&self, source: &SourceRecord) -> Result<SourceSnapshot, EditableAssetError> {
        let logical_path = SourceSnapshot::new(&source.uri, Vec::new())
            .map_err(|_| EditableAssetError::InvalidSourcePath)?
            .logical_path()
            .to_owned();
        let root = fs::canonicalize(&self.root)
            .map_err(|error| io_error("canonicalize source root", error))?;
        let resolved = fs::canonicalize(self.root.join(&logical_path))
            .map_err(|error| io_error("resolve source", error))?;
        if !resolved.starts_with(&root) {
            return Err(EditableAssetError::SourceOutsideRoot);
        }
        ensure_regular_file(&resolved)?;

        let file = File::open(&resolved).map_err(|error| io_error("open source", error))?;
        let length = file
            .metadata()
            .map_err(|error| io_error("inspect source", error))?
            .len();
        if length > MAX_EXTERNAL_SOURCE_BYTES as u64 {
            return Err(EditableAssetError::FieldTooLarge {
                field: "external source",
                limit: MAX_EXTERNAL_SOURCE_BYTES,
                actual: usize::try_from(length).unwrap_or(usize::MAX),
            });
        }
        let mut bytes = Vec::with_capacity(usize::try_from(length).unwrap_or(0));
        file.take(MAX_EXTERNAL_SOURCE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| io_error("read source", error))?;
        if bytes.len() > MAX_EXTERNAL_SOURCE_BYTES {
            return Err(EditableAssetError::FieldTooLarge {
                field: "external source",
                limit: MAX_EXTERNAL_SOURCE_BYTES,
                actual: bytes.len(),
            });
        }
        SourceSnapshot::new(&logical_path, bytes).map_err(|_| EditableAssetError::InvalidSourcePath)
    }
}

impl SourceRecord {
    /// Returns a validated external-only record.
    pub fn external(
        uri: impl Into<String>,
        format: impl Into<String>,
        content_hash: ContentHash,
    ) -> Result<Self, EditableAssetError> {
        let record = Self {
            uri: uri.into(),
            format: format.into(),
            content_hash,
            embedded_source: None,
        };
        record.validate()?;
        Ok(record)
    }

    /// Returns a self-contained record and derives its source content hash.
    pub fn embedded(
        uri: impl Into<String>,
        format: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Result<Self, EditableAssetError> {
        let record = Self {
            uri: uri.into(),
            format: format.into(),
            content_hash: ContentHash::digest(&bytes),
            embedded_source: Some(bytes),
        };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> Result<(), EditableAssetError> {
        validate_text(&self.uri, MAX_SOURCE_URI_BYTES, "source URI")?;
        validate_text(&self.format, MAX_SOURCE_FORMAT_BYTES, "source format")?;
        if let Some(bytes) = &self.embedded_source {
            if bytes.len() > MAX_EMBEDDED_SOURCE_BYTES {
                return Err(EditableAssetError::FieldTooLarge {
                    field: "embedded source",
                    limit: MAX_EMBEDDED_SOURCE_BYTES,
                    actual: bytes.len(),
                });
            }
            if ContentHash::digest(bytes) != self.content_hash {
                return Err(EditableAssetError::EmbeddedSourceHashMismatch);
            }
        }
        Ok(())
    }
}

/// Decoded canonical editable package.  `kind` and `flags` are application
/// protocol fields; this generic boundary preserves them without assigning
/// project-specific gameplay meaning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableAsset {
    pub asset: AssetId,
    pub kind: u128,
    pub flags: u32,
    pub canonical_settings: Vec<u8>,
    pub source: SourceRecord,
    pub subassets: Vec<EditableSubAsset>,
    pub dependencies: Vec<EditableDependency>,
    pub subasset_references: Vec<EditableSubAssetReference>,
    pub authoring_payload: Vec<u8>,
    pub preview: Option<Vec<u8>>,
}

/// Failure while validating or decoding an editable package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditableAssetError {
    InvalidMagic,
    UnsupportedSchema(u32),
    Truncated,
    TrailingBytes,
    PackageTooLarge {
        actual: usize,
    },
    FieldTooLarge {
        field: &'static str,
        limit: usize,
        actual: usize,
    },
    InvalidText(&'static str),
    InvalidOptionTag,
    InvalidDependencyKind,
    NonCanonicalSubassets,
    DuplicateSubAssetName,
    SubAssetKindChanged {
        id: u64,
    },
    NonCanonicalSubAssetRedirects,
    UnknownRedirectSource {
        id: u64,
    },
    UnknownRedirectTarget {
        id: u64,
    },
    RedirectSourceStillPresent {
        id: u64,
    },
    RedirectTargetIsNotNew {
        id: u64,
    },
    RedirectKindMismatch {
        from: u64,
        to: u64,
    },
    ReferencedSubAssetRemoved {
        id: u64,
    },
    NonCanonicalDependencies,
    SelfDependency,
    NonCanonicalSubAssetReferences,
    MissingReferencedAsset {
        asset: AssetId,
    },
    MissingReferencedSubAsset {
        asset: AssetId,
        subasset: u64,
    },
    DuplicateWorkspaceAsset {
        asset: AssetId,
    },
    WorkspaceAssetNotFound {
        asset: AssetId,
    },
    WorkspaceEditConflict {
        asset: AssetId,
        expected: ContentHash,
        actual: ContentHash,
    },
    WorkspacePackageLimitExceeded {
        limit: usize,
    },
    InvalidWorkspaceSnapshotBudget {
        requested: u64,
        limit: u64,
    },
    WorkspaceEncodedBytesLimitExceeded {
        actual: u64,
        limit: u64,
    },
    WorkspaceEntryLimitExceeded {
        limit: usize,
    },
    WorkspaceDirectoryLimitExceeded {
        limit: usize,
    },
    WorkspaceDepthLimitExceeded {
        limit: usize,
    },
    PackageOutsideWorkspace,
    InvalidWorkspaceTransaction,
    TransactionConflict,
    EmbeddedSourceHashMismatch,
    ExternalSourceUnavailable,
    InvalidEmbeddedSourcePath,
    InvalidSourcePath,
    SourceOutsideRoot,
    ExternalSourceHashMismatch {
        expected: ContentHash,
        actual: ContentHash,
    },
    InvalidFilePath,
    NonRegularFile,
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
}

impl fmt::Display for EditableAssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => f.write_str("invalid editable asset magic"),
            Self::UnsupportedSchema(version) => {
                write!(f, "unsupported editable asset schema {version}")
            }
            Self::Truncated => f.write_str("truncated editable asset package"),
            Self::TrailingBytes => f.write_str("editable asset package has trailing bytes"),
            Self::PackageTooLarge { actual } => write!(
                f,
                "editable asset package exceeds {MAX_EDITABLE_ASSET_BYTES} bytes: {actual}"
            ),
            Self::FieldTooLarge {
                field,
                limit,
                actual,
            } => write!(f, "editable asset {field} exceeds {limit} bytes: {actual}"),
            Self::InvalidText(field) => {
                write!(f, "editable asset {field} is empty or contains NUL")
            }
            Self::InvalidOptionTag => f.write_str("invalid editable asset option tag"),
            Self::InvalidDependencyKind => f.write_str("invalid editable asset dependency kind"),
            Self::NonCanonicalSubassets => {
                f.write_str("editable asset subassets are not strictly ordered by id")
            }
            Self::DuplicateSubAssetName => {
                f.write_str("editable asset contains duplicate subasset names")
            }
            Self::SubAssetKindChanged { id } => {
                write!(
                    f,
                    "editable asset subasset {id} changed kind across reimport"
                )
            }
            Self::NonCanonicalSubAssetRedirects => {
                f.write_str("editable asset subasset redirects are not strictly canonical")
            }
            Self::UnknownRedirectSource { id } => {
                write!(f, "editable asset redirect source {id} does not exist")
            }
            Self::UnknownRedirectTarget { id } => {
                write!(f, "editable asset redirect target {id} does not exist")
            }
            Self::RedirectSourceStillPresent { id } => {
                write!(f, "editable asset redirect source {id} is still present")
            }
            Self::RedirectTargetIsNotNew { id } => {
                write!(
                    f,
                    "editable asset redirect target {id} is not a new subasset"
                )
            }
            Self::RedirectKindMismatch { from, to } => {
                write!(
                    f,
                    "editable asset redirect {from} -> {to} changes subasset kind"
                )
            }
            Self::ReferencedSubAssetRemoved { id } => {
                write!(
                    f,
                    "editable asset referenced subasset {id} was removed without redirect"
                )
            }
            Self::NonCanonicalDependencies => {
                f.write_str("editable asset dependencies are not strictly canonical")
            }
            Self::SelfDependency => f.write_str("editable asset cannot depend on itself"),
            Self::NonCanonicalSubAssetReferences => {
                f.write_str("editable asset subasset references are not strictly canonical")
            }
            Self::MissingReferencedAsset { asset } => {
                write!(f, "editable asset references missing asset {}", asset.0)
            }
            Self::MissingReferencedSubAsset { asset, subasset } => {
                write!(
                    f,
                    "editable asset references missing subasset {subasset} on asset {}",
                    asset.0
                )
            }
            Self::DuplicateWorkspaceAsset { asset } => {
                write!(f, "editable workspace contains duplicate asset {}", asset.0)
            }
            Self::WorkspaceAssetNotFound { asset } => {
                write!(f, "editable workspace has no package for asset {}", asset.0)
            }
            Self::WorkspaceEditConflict {
                asset,
                expected,
                actual,
            } => write!(
                f,
                "editable workspace asset {} changed: expected {}, got {}",
                asset.0,
                expected.to_hex(),
                actual.to_hex()
            ),
            Self::WorkspacePackageLimitExceeded { limit } => {
                write!(f, "editable workspace exceeds {limit} package files")
            }
            Self::InvalidWorkspaceSnapshotBudget { requested, limit } => {
                write!(
                    f,
                    "editable workspace snapshot budget must be in 1..={limit} bytes: {requested}"
                )
            }
            Self::WorkspaceEncodedBytesLimitExceeded { actual, limit } => {
                write!(
                    f,
                    "editable workspace encoded packages exceed {limit} bytes: {actual}"
                )
            }
            Self::WorkspaceEntryLimitExceeded { limit } => {
                write!(f, "editable workspace traversal exceeds {limit} entries")
            }
            Self::WorkspaceDirectoryLimitExceeded { limit } => {
                write!(
                    f,
                    "editable workspace traversal exceeds {limit} directories"
                )
            }
            Self::WorkspaceDepthLimitExceeded { limit } => {
                write!(
                    f,
                    "editable workspace traversal exceeds directory depth {limit}"
                )
            }
            Self::PackageOutsideWorkspace => {
                f.write_str("editable package path escapes its workspace root")
            }
            Self::InvalidWorkspaceTransaction => {
                f.write_str("editable workspace transaction is malformed")
            }
            Self::TransactionConflict => {
                f.write_str("editable workspace package changed during transaction")
            }
            Self::EmbeddedSourceHashMismatch => {
                f.write_str("embedded source bytes do not match the declared hash")
            }
            Self::ExternalSourceUnavailable => {
                f.write_str("editable asset source is external and cannot form a source snapshot")
            }
            Self::InvalidEmbeddedSourcePath => {
                f.write_str("embedded source URI is not a canonical relative path")
            }
            Self::InvalidSourcePath => {
                f.write_str("external source path is not canonical and relative")
            }
            Self::SourceOutsideRoot => {
                f.write_str("external source resolves outside its provider root")
            }
            Self::ExternalSourceHashMismatch { expected, actual } => write!(
                f,
                "external source hash mismatch: expected {}, got {}",
                expected.to_hex(),
                actual.to_hex()
            ),
            Self::InvalidFilePath => f.write_str("editable asset path must name a .prism file"),
            Self::NonRegularFile => {
                f.write_str("editable asset path must be a regular non-symlink file")
            }
            Self::Io { operation, kind } => {
                write!(f, "editable asset {operation} I/O failed: {kind}")
            }
        }
    }
}

impl std::error::Error for EditableAssetError {}

impl EditableAsset {
    /// Creates a validated package object.  The supplied ordering is part of
    /// the protocol and must already be canonical.
    pub fn new(
        asset: AssetId,
        kind: u128,
        flags: u32,
        canonical_settings: Vec<u8>,
        source: SourceRecord,
        subassets: Vec<EditableSubAsset>,
        dependencies: Vec<EditableDependency>,
        subasset_references: Vec<EditableSubAssetReference>,
        authoring_payload: Vec<u8>,
        preview: Option<Vec<u8>>,
    ) -> Result<Self, EditableAssetError> {
        let value = Self {
            asset,
            kind,
            flags,
            canonical_settings,
            source,
            subassets,
            dependencies,
            subasset_references,
            authoring_payload,
            preview,
        };
        value.validate()?;
        Ok(value)
    }

    /// Encodes the package into its only accepted byte representation.
    pub fn encode(&self) -> Result<Vec<u8>, EditableAssetError> {
        let encoded_len = self.validated_encoded_len()?;
        let mut out = Vec::with_capacity(encoded_len);
        out.extend_from_slice(MAGIC);
        put_u32(&mut out, EDITABLE_ASSET_SCHEMA);
        put_u128(&mut out, self.asset.0);
        put_u128(&mut out, self.kind);
        put_u32(&mut out, self.flags);
        put_bytes(&mut out, &self.canonical_settings);
        put_string(&mut out, &self.source.uri);
        put_string(&mut out, &self.source.format);
        out.extend_from_slice(&self.source.content_hash.0);
        match &self.source.embedded_source {
            Some(bytes) => {
                out.push(1);
                put_bytes(&mut out, bytes);
            }
            None => out.push(0),
        }
        put_u32(
            &mut out,
            u32::try_from(self.subassets.len()).expect("validated count"),
        );
        for subasset in &self.subassets {
            put_u64(&mut out, subasset.id);
            put_string(&mut out, &subasset.name);
            put_u128(&mut out, subasset.kind);
        }
        put_u32(
            &mut out,
            u32::try_from(self.dependencies.len()).expect("validated count"),
        );
        for dependency in &self.dependencies {
            put_u128(&mut out, dependency.asset.0);
            out.push(dependency.kind.tag());
        }
        put_u32(
            &mut out,
            u32::try_from(self.subasset_references.len()).expect("validated count"),
        );
        for reference in &self.subasset_references {
            put_u128(&mut out, reference.asset.0);
            put_u64(&mut out, reference.subasset);
            out.push(reference.kind.tag());
        }
        put_bytes(&mut out, &self.authoring_payload);
        match &self.preview {
            Some(bytes) => {
                out.push(1);
                put_bytes(&mut out, bytes);
            }
            None => out.push(0),
        }
        if out.len() > MAX_EDITABLE_ASSET_BYTES {
            return Err(EditableAssetError::PackageTooLarge { actual: out.len() });
        }
        debug_assert_eq!(out.len(), encoded_len);
        Ok(out)
    }

    /// Strictly decodes canonical package bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, EditableAssetError> {
        if bytes.len() > MAX_EDITABLE_ASSET_BYTES {
            return Err(EditableAssetError::PackageTooLarge {
                actual: bytes.len(),
            });
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.take(MAGIC.len())? != MAGIC {
            return Err(EditableAssetError::InvalidMagic);
        }
        let schema = cursor.u32()?;
        if schema != EDITABLE_ASSET_SCHEMA {
            return Err(EditableAssetError::UnsupportedSchema(schema));
        }
        let asset = AssetId(cursor.u128()?);
        let kind = cursor.u128()?;
        let flags = cursor.u32()?;
        let canonical_settings = cursor.bytes(MAX_SETTINGS_BYTES, "settings")?;
        let uri = cursor.string(MAX_SOURCE_URI_BYTES, "source URI")?;
        let format = cursor.string(MAX_SOURCE_FORMAT_BYTES, "source format")?;
        let content_hash = ContentHash(cursor.array()?);
        let embedded_source = match cursor.byte()? {
            0 => None,
            1 => Some(cursor.bytes(MAX_EMBEDDED_SOURCE_BYTES, "embedded source")?),
            _ => return Err(EditableAssetError::InvalidOptionTag),
        };
        let subasset_count = cursor.count(MAX_SUBASSETS, "subassets")?;
        let mut subassets = Vec::with_capacity(subasset_count);
        for _ in 0..subasset_count {
            subassets.push(EditableSubAsset {
                id: cursor.u64()?,
                name: cursor.string(MAX_SUBASSET_NAME_BYTES, "subasset name")?,
                kind: cursor.u128()?,
            });
        }
        let dependency_count = cursor.count(MAX_DEPENDENCIES, "dependencies")?;
        let mut dependencies = Vec::with_capacity(dependency_count);
        for _ in 0..dependency_count {
            dependencies.push(EditableDependency {
                asset: AssetId(cursor.u128()?),
                kind: EditableDependencyKind::from_tag(cursor.byte()?)?,
            });
        }
        let subasset_reference_count =
            cursor.count(MAX_SUBASSET_REFERENCES, "subasset references")?;
        let mut subasset_references = Vec::with_capacity(subasset_reference_count);
        for _ in 0..subasset_reference_count {
            subasset_references.push(EditableSubAssetReference {
                asset: AssetId(cursor.u128()?),
                subasset: cursor.u64()?,
                kind: EditableDependencyKind::from_tag(cursor.byte()?)?,
            });
        }
        let authoring_payload = cursor.bytes(MAX_AUTHORING_PAYLOAD_BYTES, "authoring payload")?;
        let preview = match cursor.byte()? {
            0 => None,
            1 => Some(cursor.bytes(MAX_PREVIEW_BYTES, "preview")?),
            _ => return Err(EditableAssetError::InvalidOptionTag),
        };
        if !cursor.is_empty() {
            return Err(EditableAssetError::TrailingBytes);
        }
        Self::new(
            asset,
            kind,
            flags,
            canonical_settings,
            SourceRecord {
                uri,
                format,
                content_hash,
                embedded_source,
            },
            subassets,
            dependencies,
            subasset_references,
            authoring_payload,
            preview,
        )
    }

    /// Reads and strictly decodes one local `.prism` file.
    ///
    /// The file is bounded before allocation and must be a regular
    /// non-symlink file. This local API does not resolve `SourceRecord` URIs.
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self, EditableAssetError> {
        let path = path.as_ref();
        validate_editable_path(path)?;
        ensure_regular_file(path)?;

        let file = File::open(path).map_err(|error| io_error("open", error))?;
        let length = file
            .metadata()
            .map_err(|error| io_error("inspect", error))?
            .len();
        if length > MAX_EDITABLE_ASSET_BYTES as u64 {
            return Err(EditableAssetError::PackageTooLarge {
                actual: usize::try_from(length).unwrap_or(usize::MAX),
            });
        }

        let mut bytes = Vec::with_capacity(usize::try_from(length).unwrap_or(0));
        file.take(MAX_EDITABLE_ASSET_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| io_error("read", error))?;
        Self::decode(&bytes)
    }

    /// Atomically replaces one local `.prism` file with this canonical package.
    ///
    /// The parent directory must already exist. The temporary file is created
    /// beside the destination so replacement remains within one filesystem.
    /// This is a single-writer primitive; it intentionally provides no lock or
    /// multi-writer conflict policy.
    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<(), EditableAssetError> {
        let path = path.as_ref();
        validate_editable_path(path)?;
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        ensure_directory(parent)?;
        if path.exists() {
            ensure_regular_file(path)?;
        }

        let bytes = self.encode()?;
        let mut temporary = tempfile::NamedTempFile::new_in(parent)
            .map_err(|error| io_error("create temporary file", error))?;
        temporary
            .write_all(&bytes)
            .map_err(|error| io_error("write", error))?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|error| io_error("sync file", error))?;
        temporary
            .persist(path)
            .map_err(|error| io_error("replace", error.error))?;
        sync_directory(parent)?;
        Ok(())
    }

    /// Converts a self-contained package into the immutable build input.
    pub fn embedded_source_snapshot(&self) -> Result<SourceSnapshot, EditableAssetError> {
        self.source.validate()?;
        let bytes = self
            .source
            .embedded_source
            .as_ref()
            .ok_or(EditableAssetError::ExternalSourceUnavailable)?;
        SourceSnapshot::new(&self.source.uri, bytes.clone())
            .map_err(|_| EditableAssetError::InvalidEmbeddedSourcePath)
    }

    /// Resolves embedded or external source bytes and validates their content
    /// hash against the persisted source record.
    pub fn resolve_source_snapshot<P: EditableSourceProvider>(
        &self,
        provider: &P,
    ) -> Result<SourceSnapshot, EditableAssetError> {
        self.source.validate()?;
        let snapshot = match self.source.embedded_source.as_ref() {
            Some(_) => self.embedded_source_snapshot()?,
            None => provider.snapshot(&self.source)?,
        };
        let actual = snapshot.content_hash();
        if actual != self.source.content_hash {
            return Err(EditableAssetError::ExternalSourceHashMismatch {
                expected: self.source.content_hash,
                actual,
            });
        }
        Ok(snapshot)
    }

    /// Matches a newly imported directory against this package's stable IDs.
    ///
    /// A matching ID preserves identity. A name change with the same ID is a
    /// confirmed rename; an ID change is only a remove/add unless the caller
    /// supplies an explicit redirect. Referenced removals require a redirect.
    pub fn plan_subasset_reimport(
        &self,
        next_subassets: Vec<EditableSubAsset>,
        referenced_subassets: &[u64],
        redirects: &[EditableSubAssetRedirect],
    ) -> Result<EditableSubAssetReimportPlan, EditableAssetError> {
        validate_subassets(&self.subassets)?;
        validate_subassets(&next_subassets)?;

        let previous_by_id = self
            .subassets
            .iter()
            .map(|subasset| (subasset.id, subasset))
            .collect::<BTreeMap<_, _>>();
        let next_by_id = next_subassets
            .iter()
            .map(|subasset| (subasset.id, subasset))
            .collect::<BTreeMap<_, _>>();
        let redirect_by_source = validate_redirects(&previous_by_id, &next_by_id, redirects)?;
        let referenced = referenced_subassets
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let redirect_targets = redirect_by_source
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut changes = Vec::with_capacity(self.subassets.len() + next_subassets.len());

        for previous in &self.subassets {
            if let Some(next) = next_by_id.get(&previous.id) {
                if previous.kind != next.kind {
                    return Err(EditableAssetError::SubAssetKindChanged { id: previous.id });
                }
                if previous.name == next.name {
                    changes.push(EditableSubAssetChange::Retained { id: previous.id });
                } else {
                    changes.push(EditableSubAssetChange::Renamed {
                        id: previous.id,
                        previous_name: previous.name.clone(),
                        next_name: next.name.clone(),
                    });
                }
            } else if let Some(&to) = redirect_by_source.get(&previous.id) {
                changes.push(EditableSubAssetChange::Redirected {
                    from: previous.id,
                    to,
                });
            } else if referenced.contains(&previous.id) {
                return Err(EditableAssetError::ReferencedSubAssetRemoved { id: previous.id });
            } else {
                changes.push(EditableSubAssetChange::Removed {
                    subasset: previous.clone(),
                });
            }
        }

        for next in &next_subassets {
            if !previous_by_id.contains_key(&next.id) && !redirect_targets.contains(&next.id) {
                changes.push(EditableSubAssetChange::Added {
                    subasset: next.clone(),
                });
            }
        }

        Ok(EditableSubAssetReimportPlan {
            subassets: next_subassets,
            changes,
        })
    }

    /// Checks canonical fields and the total encoded size without encoding a copy.
    pub(crate) fn validate(&self) -> Result<(), EditableAssetError> {
        if self.canonical_settings.len() > MAX_SETTINGS_BYTES {
            return Err(EditableAssetError::FieldTooLarge {
                field: "settings",
                limit: MAX_SETTINGS_BYTES,
                actual: self.canonical_settings.len(),
            });
        }
        if self.authoring_payload.len() > MAX_AUTHORING_PAYLOAD_BYTES {
            return Err(EditableAssetError::FieldTooLarge {
                field: "authoring payload",
                limit: MAX_AUTHORING_PAYLOAD_BYTES,
                actual: self.authoring_payload.len(),
            });
        }
        if let Some(preview) = &self.preview {
            if preview.len() > MAX_PREVIEW_BYTES {
                return Err(EditableAssetError::FieldTooLarge {
                    field: "preview",
                    limit: MAX_PREVIEW_BYTES,
                    actual: preview.len(),
                });
            }
        }
        self.source.validate()?;
        validate_subassets(&self.subassets)?;
        if self.dependencies.len() > MAX_DEPENDENCIES {
            return Err(EditableAssetError::FieldTooLarge {
                field: "dependencies",
                limit: MAX_DEPENDENCIES,
                actual: self.dependencies.len(),
            });
        }
        for pair in self.dependencies.windows(2) {
            if pair[0] >= pair[1] {
                return Err(EditableAssetError::NonCanonicalDependencies);
            }
        }
        if self
            .dependencies
            .iter()
            .any(|dependency| dependency.asset == self.asset)
        {
            return Err(EditableAssetError::SelfDependency);
        }
        if self.subasset_references.len() > MAX_SUBASSET_REFERENCES {
            return Err(EditableAssetError::FieldTooLarge {
                field: "subasset references",
                limit: MAX_SUBASSET_REFERENCES,
                actual: self.subasset_references.len(),
            });
        }
        for pair in self.subasset_references.windows(2) {
            if pair[0] >= pair[1] {
                return Err(EditableAssetError::NonCanonicalSubAssetReferences);
            }
        }
        let actual = self.encoded_len();
        if actual > MAX_EDITABLE_ASSET_BYTES {
            return Err(EditableAssetError::PackageTooLarge { actual });
        }
        Ok(())
    }

    /// Validates an owned package and returns its exact canonical encoded size.
    pub(crate) fn validated_encoded_len(&self) -> Result<usize, EditableAssetError> {
        self.validate()?;
        Ok(self.encoded_len())
    }

    // Mirrors the canonical encoder, including every length prefix and option
    // tag. Saturation also makes this safe for unvalidated caller-owned fields.
    fn encoded_len(&self) -> usize {
        let mut length = MAGIC.len() + 4 + 16 + 16 + 4 + 32 + 1 + 4 + 4 + 4 + 1;
        for bytes in [
            self.canonical_settings.len(),
            self.source.uri.len(),
            self.source.format.len(),
            self.authoring_payload.len(),
        ] {
            length = length.saturating_add(4).saturating_add(bytes);
        }
        if let Some(bytes) = &self.source.embedded_source {
            length = length.saturating_add(4).saturating_add(bytes.len());
        }
        for subasset in &self.subassets {
            length = length
                .saturating_add(8 + 4 + 16)
                .saturating_add(subasset.name.len());
        }
        length = length.saturating_add(self.dependencies.len().saturating_mul(16 + 1));
        length = length.saturating_add(self.subasset_references.len().saturating_mul(16 + 8 + 1));
        if let Some(bytes) = &self.preview {
            length = length.saturating_add(4).saturating_add(bytes.len());
        }
        length
    }
}

impl EditableWorkspace {
    /// Opens an existing, non-symlink workspace and recovers any interrupted
    /// local publication before exposing it to callers.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, EditableAssetError> {
        let supplied_root = root.into();
        ensure_directory(&supplied_root)?;
        let root = fs::canonicalize(&supplied_root)
            .map_err(|error| io_error("canonicalize workspace root", error))?;
        ensure_directory(&root)?;
        let workspace = Self { root };
        workspace.with_lock(|| recover_workspace_transactions(&workspace.root))?;
        Ok(workspace)
    }

    /// Returns the canonical local workspace root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Captures owned canonical packages under the workspace lock after recovery.
    ///
    /// The lock is released before this method returns; callers can cook these
    /// immutable inputs without holding it. Later package replacement cannot
    /// change the returned assets. Writers must cooperate with the workspace lock;
    /// direct filesystem writes are not a transactional publication mechanism.
    /// Hard and editor-only subasset references must resolve within this captured
    /// workspace. Soft references may name absent assets; if their target asset
    /// is present, its referenced subasset must exist.
    ///
    /// `max_encoded_bytes` must be positive and at most
    /// [`MAX_EDITABLE_WORKSPACE_ENCODED_BYTES`] (512 MiB). Metadata for the entire
    /// scan is checked against that aggregate budget before allocating encoded
    /// package buffers or decoding packages. Decoded objects and scratch storage
    /// have additional bounded overhead; this is an encoded-byte budget.
    ///
    /// Traversal is iterative and accepts at most 65,536 package files, 262,144
    /// entries (including unrelated files and symlinks), 16,384 directories
    /// including the root, and 64 directory levels below the root. Symlinks are
    /// skipped. The reserved transaction subtree is recovered separately rather
    /// than imported. These same traversal and hard byte caps apply to reimport.
    pub fn snapshot(
        &self,
        max_encoded_bytes: u64,
    ) -> Result<BTreeMap<AssetId, EditableAsset>, EditableAssetError> {
        if max_encoded_bytes == 0 || max_encoded_bytes > MAX_EDITABLE_WORKSPACE_ENCODED_BYTES {
            return Err(EditableAssetError::InvalidWorkspaceSnapshotBudget {
                requested: max_encoded_bytes,
                limit: MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
            });
        }
        self.with_lock(|| {
            recover_workspace_transactions(&self.root)?;
            let packages =
                scan_workspace_packages(&self.root, WorkspaceScanLimits::new(max_encoded_bytes))?;
            validate_workspace_reference_graph(&packages)?;
            Ok(packages
                .into_iter()
                .map(|(id, package)| (id, package.asset))
                .collect())
        })
    }

    /// Compares and saves replacements in one recoverable local transaction.
    ///
    /// All expected package hashes, replacement encodings, aggregate encoded
    /// byte bounds and the complete candidate subasset-reference graph are
    /// validated before staging or replacing any package. Duplicate identities,
    /// unknown packages and stale edits fail the entire batch. Callers supply
    /// both sides of a reference change together; opaque payloads are never
    /// guessed or rewritten by this method. The edits are consumed so large
    /// authoring payloads move into the candidate workspace without cloning.
    ///
    /// Publication uses the same workspace lock and prepare/commit recovery as
    /// reimport. Cooperating readers see one revision through `snapshot`; direct
    /// filesystem readers may observe replacement in progress. The byte bound
    /// covers each old/new encoded workspace, not total process heap usage.
    /// A no-op batch does not create a transaction.
    pub fn save(
        &self,
        edits: Vec<EditableWorkspaceEdit>,
        max_encoded_bytes: u64,
    ) -> Result<EditableWorkspaceSave, EditableAssetError> {
        if max_encoded_bytes == 0 || max_encoded_bytes > MAX_EDITABLE_WORKSPACE_ENCODED_BYTES {
            return Err(EditableAssetError::InvalidWorkspaceSnapshotBudget {
                requested: max_encoded_bytes,
                limit: MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
            });
        }
        if edits.len() > MAX_WORKSPACE_PACKAGES {
            return Err(EditableAssetError::WorkspacePackageLimitExceeded {
                limit: MAX_WORKSPACE_PACKAGES,
            });
        }
        let mut ordered = BTreeMap::new();
        let mut replacement_bytes = 0_u64;
        for edit in edits {
            replacement_bytes =
                replacement_bytes.saturating_add(edit.replacement.validated_encoded_len()? as u64);
            if replacement_bytes > max_encoded_bytes {
                return Err(EditableAssetError::WorkspaceEncodedBytesLimitExceeded {
                    actual: replacement_bytes,
                    limit: max_encoded_bytes,
                });
            }
            let asset = edit.replacement.asset;
            if ordered.insert(asset, edit).is_some() {
                return Err(EditableAssetError::DuplicateWorkspaceAsset { asset });
            }
        }
        self.with_lock(|| {
            recover_workspace_transactions(&self.root)?;
            let mut packages =
                scan_workspace_packages(&self.root, WorkspaceScanLimits::new(max_encoded_bytes))?;
            let mut next_bytes = packages
                .values()
                .map(|p| p.encoded.len() as u64)
                .sum::<u64>();
            // Compare the whole batch before replacing any candidate or encoding
            // any output. Sorted identity order makes rejection deterministic.
            for (&asset, edit) in &ordered {
                let previous = packages
                    .get(&asset)
                    .ok_or(EditableAssetError::WorkspaceAssetNotFound { asset })?;
                let actual = ContentHash::digest(&previous.encoded);
                if actual != edit.expected_hash {
                    return Err(EditableAssetError::WorkspaceEditConflict {
                        asset,
                        expected: edit.expected_hash,
                        actual,
                    });
                }
                next_bytes = next_bytes - previous.encoded.len() as u64
                    + edit.replacement.encoded_len() as u64;
            }
            if next_bytes > max_encoded_bytes {
                return Err(EditableAssetError::WorkspaceEncodedBytesLimitExceeded {
                    actual: next_bytes,
                    limit: max_encoded_bytes,
                });
            }
            for (&asset, edit) in ordered.iter_mut() {
                // Swapping moves the potentially large payload rather than
                // cloning it; the previous value drops before output encoding.
                std::mem::swap(
                    &mut packages.get_mut(&asset).expect("compared package").asset,
                    &mut edit.replacement,
                );
            }
            let edited_ids = ordered.keys().copied().collect::<Vec<_>>();
            drop(ordered);
            validate_workspace_reference_graph(&packages)?;
            let mut records = Vec::new();
            let mut content_hashes = BTreeMap::new();
            let mut updated_assets = Vec::new();
            for asset in edited_ids {
                let package = packages.get_mut(&asset).expect("compared package");
                let next = package.asset.encode()?;
                content_hashes.insert(asset, ContentHash::digest(&next));
                if next != package.encoded {
                    updated_assets.push(asset);
                    records.push(WorkspaceTransactionRecord {
                        relative_path: package.relative_path.clone(),
                        old_bytes: std::mem::take(&mut package.encoded),
                        new_bytes: next,
                    });
                }
            }
            // Validation no longer needs decoded packages or unchanged bytes.
            drop(packages);
            publish_workspace_transaction(&self.root, &records)?;
            Ok(EditableWorkspaceSave {
                updated_assets,
                content_hashes,
            })
        })
    }

    /// Reimports one package and atomically publishes every explicit reference
    /// update required by caller-provided stable subasset redirects.
    pub fn reimport<P, I, R>(
        &self,
        package_path: impl AsRef<Path>,
        provider: &P,
        importer: &I,
        payload_rewriter: &R,
        redirects: &[EditableSubAssetRedirect],
    ) -> Result<EditableWorkspaceReimport, EditableAssetError>
    where
        P: EditableSourceProvider,
        I: EditableReimporter,
        R: EditablePayloadReferenceRewriter,
    {
        let package_path = package_path.as_ref().to_path_buf();
        self.with_lock(|| {
            recover_workspace_transactions(&self.root)?;
            let mut packages = scan_workspace_packages(
                &self.root,
                WorkspaceScanLimits::new(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES),
            )?;
            let target_path = validate_workspace_package_path(&package_path)?;
            let target_asset = packages
                .values()
                .find(|package| package.relative_path == target_path)
                .map(|package| package.asset.asset)
                .ok_or(EditableAssetError::PackageOutsideWorkspace)?;
            let previous = packages
                .get(&target_asset)
                .expect("target was found in package map")
                .asset
                .clone();

            let snapshot = capture_reimport_source(&previous, provider)?;
            let output = importer.reimport(&previous, &snapshot)?;
            let referenced_subassets = packages
                .values()
                .flat_map(|package| package.asset.subasset_references.iter())
                .filter(|reference| reference.asset == target_asset)
                .map(|reference| reference.subasset)
                .collect::<Vec<_>>();
            let plan = previous.plan_subasset_reimport(
                output.subassets,
                &referenced_subassets,
                redirects,
            )?;
            let source = SourceRecord {
                uri: previous.source.uri.clone(),
                format: previous.source.format.clone(),
                content_hash: snapshot.content_hash(),
                embedded_source: previous
                    .source
                    .embedded_source
                    .as_ref()
                    .map(|_| snapshot.bytes().to_vec()),
            };
            let next = EditableAsset::new(
                previous.asset,
                previous.kind,
                previous.flags,
                output.canonical_settings,
                source,
                plan.subassets.clone(),
                output.dependencies,
                output.subasset_references,
                output.authoring_payload,
                output.preview,
            )?;
            packages
                .get_mut(&target_asset)
                .expect("target was found in package map")
                .asset = next;

            let redirect_by_source = redirects
                .iter()
                .map(|redirect| (redirect.from, redirect.to))
                .collect::<BTreeMap<_, _>>();
            if !redirect_by_source.is_empty() {
                for package in packages.values_mut() {
                    let rewrites = package
                        .asset
                        .subasset_references
                        .iter()
                        .filter_map(|reference| {
                            (reference.asset == target_asset)
                                .then(|| redirect_by_source.get(&reference.subasset))
                                .flatten()
                                .map(|&to| EditableSubAssetReferenceRewrite {
                                    asset: target_asset,
                                    from: reference.subasset,
                                    to,
                                })
                        })
                        .collect::<Vec<_>>();
                    if rewrites.is_empty() {
                        continue;
                    }

                    let authoring_payload =
                        payload_rewriter.rewrite_authoring_payload(&package.asset, &rewrites)?;
                    for reference in &mut package.asset.subasset_references {
                        if reference.asset == target_asset {
                            if let Some(&to) = redirect_by_source.get(&reference.subasset) {
                                reference.subasset = to;
                            }
                        }
                    }
                    package.asset.authoring_payload = authoring_payload;
                    package.asset.validate()?;
                }
            }

            validate_workspace_reference_graph(&packages)?;
            let mut changes = Vec::new();
            let mut updated_assets = Vec::new();
            for package in packages.values() {
                let next = package.asset.encode()?;
                if next != package.encoded {
                    updated_assets.push(package.asset.asset);
                    changes.push(WorkspaceTransactionRecord {
                        relative_path: package.relative_path.clone(),
                        old_bytes: package.encoded.clone(),
                        new_bytes: next,
                    });
                }
            }
            publish_workspace_transaction(&self.root, &changes)?;

            Ok(EditableWorkspaceReimport {
                plan,
                updated_assets,
            })
        })
    }

    fn with_lock<T>(
        &self,
        operation: impl FnOnce() -> Result<T, EditableAssetError>,
    ) -> Result<T, EditableAssetError> {
        let lock_path = self.root.join(".prism-editable.lock");
        if lock_path.exists() {
            ensure_regular_file(&lock_path)?;
        }
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|error| io_error("open workspace lock", error))?;
        ensure_regular_file(&lock_path)?;
        lock.lock_exclusive()
            .map_err(|error| io_error("lock workspace", error))?;
        let result = operation();
        let unlock = FileExt::unlock(&lock).map_err(|error| io_error("unlock workspace", error));
        match (result, unlock) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }
}

#[derive(Clone, Debug)]
struct WorkspacePackage {
    relative_path: PathBuf,
    asset: EditableAsset,
    encoded: Vec<u8>,
}

#[derive(Clone, Debug)]
struct WorkspaceTransactionRecord {
    relative_path: PathBuf,
    old_bytes: Vec<u8>,
    new_bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct WorkspaceTransactionManifestRecord {
    relative_path: PathBuf,
    old_hash: ContentHash,
    new_hash: ContentHash,
}

fn capture_reimport_source<P: EditableSourceProvider>(
    asset: &EditableAsset,
    provider: &P,
) -> Result<SourceSnapshot, EditableAssetError> {
    asset.source.validate()?;
    let snapshot = match asset.source.embedded_source.as_ref() {
        Some(_) => asset.embedded_source_snapshot()?,
        None => provider.snapshot(&asset.source)?,
    };
    if snapshot.logical_path() != asset.source.uri {
        return Err(EditableAssetError::InvalidSourcePath);
    }
    Ok(snapshot)
}

fn validate_workspace_package_path(path: &Path) -> Result<PathBuf, EditableAssetError> {
    validate_editable_path(path)?;
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(EditableAssetError::PackageOutsideWorkspace);
    }
    Ok(path.to_path_buf())
}

#[derive(Clone, Copy)]
struct WorkspaceScanLimits {
    encoded_bytes: u64,
    packages: usize,
    entries: usize,
    directories: usize,
    depth: usize,
}

impl WorkspaceScanLimits {
    const fn new(encoded_bytes: u64) -> Self {
        Self {
            encoded_bytes,
            packages: MAX_WORKSPACE_PACKAGES,
            entries: MAX_WORKSPACE_ENTRIES,
            directories: MAX_WORKSPACE_DIRECTORIES,
            depth: MAX_WORKSPACE_DIRECTORY_DEPTH,
        }
    }
}

fn scan_workspace_packages(
    root: &Path,
    limits: WorkspaceScanLimits,
) -> Result<BTreeMap<AssetId, WorkspacePackage>, EditableAssetError> {
    // Finish the aggregate metadata preflight before allocating any package
    // bytes. The path list is itself bounded during traversal, not after it.
    let mut paths = scan_workspace_directory(root, limits)?;
    paths.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let mut packages = BTreeMap::new();
    for (path, length) in paths {
        let (asset, encoded) = read_editable_package_bytes_at_length(&path, Some(length))?;
        let relative_path = path
            .strip_prefix(root)
            .map_err(|_| EditableAssetError::PackageOutsideWorkspace)?
            .to_path_buf();
        validate_workspace_package_path(&relative_path)?;
        let id = asset.asset;
        if packages
            .insert(
                id,
                WorkspacePackage {
                    relative_path,
                    asset,
                    encoded,
                },
            )
            .is_some()
        {
            return Err(EditableAssetError::DuplicateWorkspaceAsset { asset: id });
        }
    }
    Ok(packages)
}

fn scan_workspace_directory(
    root: &Path,
    limits: WorkspaceScanLimits,
) -> Result<Vec<(PathBuf, u64)>, EditableAssetError> {
    ensure_directory(root)?;
    if limits.directories == 0 {
        return Err(EditableAssetError::WorkspaceDirectoryLimitExceeded { limit: 0 });
    }
    // Streaming depth-first traversal holds at most depth + 1 read_dir handles.
    // No directory-wide entry collection or recursive call stack is needed.
    let mut stack = vec![fs::read_dir(root).map_err(|error| io_error("scan workspace", error))?];
    let mut paths = Vec::new();
    let mut entries_seen = 0;
    let mut directories_seen = 1;
    let mut encoded_bytes = 0_u64;
    while let Some(entries) = stack.last_mut() {
        let Some(entry) = entries.next() else {
            stack.pop();
            continue;
        };
        if entries_seen == limits.entries {
            return Err(EditableAssetError::WorkspaceEntryLimitExceeded {
                limit: limits.entries,
            });
        }
        entries_seen += 1;
        let entry = entry.map_err(|error| io_error("scan workspace", error))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| io_error("inspect workspace entry", error))?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            if stack.len() == 1 && entry.file_name() == EDITABLE_TRANSACTION_DIRECTORY {
                continue;
            }
            // stack.len() is the depth that this child directory would have.
            if stack.len() > limits.depth {
                return Err(EditableAssetError::WorkspaceDepthLimitExceeded {
                    limit: limits.depth,
                });
            }
            if directories_seen == limits.directories {
                return Err(EditableAssetError::WorkspaceDirectoryLimitExceeded {
                    limit: limits.directories,
                });
            }
            directories_seen += 1;
            ensure_directory(&path)?;
            stack.push(fs::read_dir(&path).map_err(|error| io_error("scan workspace", error))?);
        } else if metadata.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension == "prism")
        {
            if paths.len() == limits.packages {
                return Err(EditableAssetError::WorkspacePackageLimitExceeded {
                    limit: limits.packages,
                });
            }
            let length = metadata.len();
            if length > MAX_EDITABLE_ASSET_BYTES as u64 {
                return Err(EditableAssetError::PackageTooLarge {
                    actual: usize::try_from(length).unwrap_or(usize::MAX),
                });
            }
            encoded_bytes = encoded_bytes.saturating_add(length);
            if encoded_bytes > limits.encoded_bytes {
                return Err(EditableAssetError::WorkspaceEncodedBytesLimitExceeded {
                    actual: encoded_bytes,
                    limit: limits.encoded_bytes,
                });
            }
            paths.push((path, length));
        }
    }
    Ok(paths)
}

fn read_editable_package_bytes(
    path: &Path,
) -> Result<(EditableAsset, Vec<u8>), EditableAssetError> {
    read_editable_package_bytes_at_length(path, None)
}

fn read_editable_package_bytes_at_length(
    path: &Path,
    expected_length: Option<u64>,
) -> Result<(EditableAsset, Vec<u8>), EditableAssetError> {
    validate_editable_path(path)?;
    ensure_regular_file(path)?;
    let mut file = File::open(path).map_err(|error| io_error("open package", error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error("inspect package", error))?
        .len();
    if length > MAX_EDITABLE_ASSET_BYTES as u64 {
        return Err(EditableAssetError::PackageTooLarge {
            actual: usize::try_from(length).unwrap_or(usize::MAX),
        });
    }
    if expected_length.is_some_and(|expected| length != expected) {
        return Err(EditableAssetError::TransactionConflict);
    }
    // A fixed-size read cannot grow its allocation if a noncooperating writer
    // extends the file after metadata preflight. Probe EOF using stack storage.
    let mut encoded = vec![0; length as usize];
    file.read_exact(&mut encoded).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            EditableAssetError::TransactionConflict
        } else {
            io_error("read package", error)
        }
    })?;
    let mut trailing = [0; 1];
    if file
        .read(&mut trailing)
        .map_err(|error| io_error("read package", error))?
        != 0
    {
        return Err(EditableAssetError::TransactionConflict);
    }
    let asset = EditableAsset::decode(&encoded)?;
    if asset.encode()? != encoded {
        return Err(EditableAssetError::InvalidWorkspaceTransaction);
    }
    Ok((asset, encoded))
}

fn validate_workspace_reference_graph(
    packages: &BTreeMap<AssetId, WorkspacePackage>,
) -> Result<(), EditableAssetError> {
    for package in packages.values() {
        for reference in &package.asset.subasset_references {
            let target = match packages.get(&reference.asset) {
                Some(target) => target,
                None if reference.kind == EditableDependencyKind::Soft => continue,
                None => {
                    return Err(EditableAssetError::MissingReferencedAsset {
                        asset: reference.asset,
                    });
                }
            };
            if target
                .asset
                .subassets
                .binary_search_by_key(&reference.subasset, |subasset| subasset.id)
                .is_err()
            {
                return Err(EditableAssetError::MissingReferencedSubAsset {
                    asset: reference.asset,
                    subasset: reference.subasset,
                });
            }
        }
    }
    Ok(())
}

fn workspace_transactions_root(root: &Path) -> PathBuf {
    root.join(EDITABLE_TRANSACTION_DIRECTORY)
}

fn ensure_workspace_transactions_root(root: &Path) -> Result<PathBuf, EditableAssetError> {
    let transactions = workspace_transactions_root(root);
    if transactions.exists() {
        ensure_directory(&transactions)?;
    } else {
        fs::create_dir(&transactions)
            .map_err(|error| io_error("create transaction root", error))?;
        sync_directory(root)?;
    }
    Ok(transactions)
}

fn publish_workspace_transaction(
    root: &Path,
    records: &[WorkspaceTransactionRecord],
) -> Result<(), EditableAssetError> {
    if records.is_empty() {
        return Ok(());
    }
    let transaction_root = ensure_workspace_transactions_root(root)?;
    let transaction = create_workspace_transaction(&transaction_root, records)?;
    let apply = apply_workspace_transaction(root, &transaction, records);
    if let Err(error) = apply {
        let rollback = recover_workspace_transaction(root, &transaction, false);
        if rollback.is_ok() {
            let _ = fs::remove_dir_all(&transaction);
            let _ = sync_directory(&transaction_root);
        }
        return Err(error);
    }

    if let Err(error) = write_new_file(
        &transaction.join("committed"),
        EDITABLE_TRANSACTION_COMMITTED,
    )
    .and_then(|()| sync_directory(&transaction))
    {
        let rollback = recover_workspace_transaction(root, &transaction, false);
        if rollback.is_ok() {
            let _ = fs::remove_dir_all(&transaction);
            let _ = sync_directory(&transaction_root);
        }
        return Err(error);
    }
    let _ = fs::remove_dir_all(&transaction);
    let _ = sync_directory(&transaction_root);
    Ok(())
}

fn create_workspace_transaction(
    transaction_root: &Path,
    records: &[WorkspaceTransactionRecord],
) -> Result<PathBuf, EditableAssetError> {
    if records.len() > MAX_WORKSPACE_PACKAGES {
        return Err(EditableAssetError::WorkspacePackageLimitExceeded {
            limit: MAX_WORKSPACE_PACKAGES,
        });
    }
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = EDITABLE_TRANSACTION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let transaction = transaction_root.join(format!("{timestamp:032x}-{counter:016x}"));
    fs::create_dir(&transaction).map_err(|error| io_error("create transaction", error))?;

    let stage = (|| {
        for (index, record) in records.iter().enumerate() {
            validate_workspace_package_path(&record.relative_path)?;
            write_new_file(
                &transaction.join(transaction_blob_name(index, "old")),
                &record.old_bytes,
            )?;
            write_new_file(
                &transaction.join(transaction_blob_name(index, "new")),
                &record.new_bytes,
            )?;
        }
        let manifest = encode_workspace_transaction_manifest(records)?;
        write_new_file(&transaction.join("manifest"), &manifest)?;
        sync_directory(&transaction)?;
        write_new_file(&transaction.join("prepared"), EDITABLE_TRANSACTION_PREPARED)?;
        sync_directory(&transaction)
    })();
    match stage {
        Ok(()) => Ok(transaction),
        Err(error) => {
            let _ = fs::remove_dir_all(&transaction);
            let _ = sync_directory(transaction_root);
            Err(error)
        }
    }
}

fn apply_workspace_transaction(
    root: &Path,
    transaction: &Path,
    records: &[WorkspaceTransactionRecord],
) -> Result<(), EditableAssetError> {
    for record in records {
        let path = root.join(&record.relative_path);
        let (_, current) = read_editable_package_bytes(&path)?;
        if current != record.old_bytes {
            return Err(EditableAssetError::TransactionConflict);
        }
        replace_file_bytes(&path, &record.new_bytes)?;
    }
    sync_directory(transaction)?;
    Ok(())
}

fn recover_workspace_transactions(root: &Path) -> Result<(), EditableAssetError> {
    let transaction_root = workspace_transactions_root(root);
    if !transaction_root.exists() {
        return Ok(());
    }
    ensure_directory(&transaction_root)?;
    let mut entries = Vec::new();
    for entry in
        fs::read_dir(&transaction_root).map_err(|error| io_error("scan transactions", error))?
    {
        if entries.len() == MAX_WORKSPACE_DIRECTORIES {
            return Err(EditableAssetError::WorkspaceDirectoryLimitExceeded {
                limit: MAX_WORKSPACE_DIRECTORIES,
            });
        }
        entries.push(entry.map_err(|error| io_error("scan transactions", error))?);
    }
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let transaction = entry.path();
        let metadata = fs::symlink_metadata(&transaction)
            .map_err(|error| io_error("inspect transaction", error))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(EditableAssetError::InvalidWorkspaceTransaction);
        }
        let prepared = transaction.join("prepared");
        if !prepared.exists() {
            fs::remove_dir_all(&transaction)
                .map_err(|error| io_error("remove incomplete transaction", error))?;
            sync_directory(&transaction_root)?;
            continue;
        }
        validate_transaction_marker(&prepared, EDITABLE_TRANSACTION_PREPARED)?;
        let committed = transaction.join("committed");
        let is_committed = if committed.exists() {
            validate_transaction_marker(&committed, EDITABLE_TRANSACTION_COMMITTED)?;
            true
        } else {
            false
        };
        recover_workspace_transaction(root, &transaction, is_committed)?;
        fs::remove_dir_all(&transaction).map_err(|error| io_error("remove transaction", error))?;
        sync_directory(&transaction_root)?;
    }
    Ok(())
}

fn recover_workspace_transaction(
    root: &Path,
    transaction: &Path,
    committed: bool,
) -> Result<(), EditableAssetError> {
    let records = decode_workspace_transaction_manifest(&read_transaction_file(
        &transaction.join("manifest"),
        MAX_EDITABLE_TRANSACTION_MANIFEST_BYTES,
    )?)?;
    for (index, record) in records.iter().enumerate() {
        let path = root.join(&record.relative_path);
        let old_bytes = read_transaction_blob(transaction, index, "old", record.old_hash)?;
        let new_bytes = read_transaction_blob(transaction, index, "new", record.new_hash)?;
        let (_, current) = read_editable_package_bytes(&path)
            .map_err(|_| EditableAssetError::TransactionConflict)?;
        let desired = if committed { &new_bytes } else { &old_bytes };
        let alternate = if committed { &old_bytes } else { &new_bytes };
        if current != *desired && current != *alternate {
            return Err(EditableAssetError::TransactionConflict);
        }
        if current != *desired {
            replace_file_bytes(&path, desired)?;
        }
    }
    Ok(())
}

fn transaction_blob_name(index: usize, suffix: &str) -> String {
    format!("{index:08x}.{suffix}")
}

fn read_transaction_file(path: &Path, limit: usize) -> Result<Vec<u8>, EditableAssetError> {
    ensure_regular_file(path).map_err(|_| EditableAssetError::InvalidWorkspaceTransaction)?;
    let file = File::open(path).map_err(|error| io_error("open transaction", error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error("inspect transaction", error))?
        .len();
    if length > limit as u64 {
        return Err(EditableAssetError::InvalidWorkspaceTransaction);
    }
    let mut bytes = Vec::with_capacity(usize::try_from(length).unwrap_or(0));
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error("read transaction", error))?;
    if bytes.len() > limit {
        return Err(EditableAssetError::InvalidWorkspaceTransaction);
    }
    Ok(bytes)
}

fn validate_transaction_marker(path: &Path, expected: &[u8]) -> Result<(), EditableAssetError> {
    if read_transaction_file(path, MAX_EDITABLE_TRANSACTION_MARKER_BYTES)? != expected {
        return Err(EditableAssetError::InvalidWorkspaceTransaction);
    }
    Ok(())
}

fn read_transaction_blob(
    transaction: &Path,
    index: usize,
    suffix: &str,
    expected_hash: ContentHash,
) -> Result<Vec<u8>, EditableAssetError> {
    let bytes = read_transaction_file(
        &transaction.join(transaction_blob_name(index, suffix)),
        MAX_EDITABLE_ASSET_BYTES,
    )?;
    if bytes.len() > MAX_EDITABLE_ASSET_BYTES || ContentHash::digest(&bytes) != expected_hash {
        return Err(EditableAssetError::InvalidWorkspaceTransaction);
    }
    Ok(bytes)
}

fn encode_workspace_transaction_manifest(
    records: &[WorkspaceTransactionRecord],
) -> Result<Vec<u8>, EditableAssetError> {
    let mut out = Vec::new();
    out.extend_from_slice(EDITABLE_TRANSACTION_MAGIC);
    put_u32(&mut out, EDITABLE_TRANSACTION_SCHEMA);
    put_u32(
        &mut out,
        u32::try_from(records.len()).map_err(|_| {
            EditableAssetError::WorkspacePackageLimitExceeded {
                limit: MAX_WORKSPACE_PACKAGES,
            }
        })?,
    );
    for record in records {
        let relative_path = validate_workspace_package_path(&record.relative_path)?;
        let path = relative_path
            .to_str()
            .ok_or(EditableAssetError::InvalidWorkspaceTransaction)?;
        put_string(&mut out, path);
        out.extend_from_slice(&ContentHash::digest(&record.old_bytes).0);
        out.extend_from_slice(&ContentHash::digest(&record.new_bytes).0);
    }
    Ok(out)
}

fn decode_workspace_transaction_manifest(
    bytes: &[u8],
) -> Result<Vec<WorkspaceTransactionManifestRecord>, EditableAssetError> {
    let mut cursor = Cursor::new(bytes);
    if cursor
        .take(EDITABLE_TRANSACTION_MAGIC.len())
        .map_err(|_| EditableAssetError::InvalidWorkspaceTransaction)?
        != EDITABLE_TRANSACTION_MAGIC
    {
        return Err(EditableAssetError::InvalidWorkspaceTransaction);
    }
    let schema = cursor
        .u32()
        .map_err(|_| EditableAssetError::InvalidWorkspaceTransaction)?;
    if schema != EDITABLE_TRANSACTION_SCHEMA {
        return Err(EditableAssetError::InvalidWorkspaceTransaction);
    }
    let count = cursor
        .count(MAX_WORKSPACE_PACKAGES, "workspace packages")
        .map_err(|_| EditableAssetError::InvalidWorkspaceTransaction)?;
    let mut paths = BTreeSet::new();
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let path = cursor
            .string(MAX_SOURCE_URI_BYTES, "workspace package path")
            .map_err(|_| EditableAssetError::InvalidWorkspaceTransaction)?;
        let relative_path = validate_workspace_package_path(Path::new(&path))
            .map_err(|_| EditableAssetError::InvalidWorkspaceTransaction)?;
        if !paths.insert(relative_path.clone()) {
            return Err(EditableAssetError::InvalidWorkspaceTransaction);
        }
        records.push(WorkspaceTransactionManifestRecord {
            relative_path,
            old_hash: ContentHash(
                cursor
                    .array()
                    .map_err(|_| EditableAssetError::InvalidWorkspaceTransaction)?,
            ),
            new_hash: ContentHash(
                cursor
                    .array()
                    .map_err(|_| EditableAssetError::InvalidWorkspaceTransaction)?,
            ),
        });
    }
    if !cursor.is_empty() {
        return Err(EditableAssetError::InvalidWorkspaceTransaction);
    }
    Ok(records)
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), EditableAssetError> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| io_error("create transaction file", error))?;
    file.write_all(bytes)
        .map_err(|error| io_error("write transaction file", error))?;
    file.sync_all()
        .map_err(|error| io_error("sync transaction file", error))
}

fn replace_file_bytes(path: &Path, bytes: &[u8]) -> Result<(), EditableAssetError> {
    validate_editable_path(path)?;
    ensure_regular_file(path)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or(EditableAssetError::PackageOutsideWorkspace)?;
    ensure_directory(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| io_error("create package replacement", error))?;
    temporary
        .write_all(bytes)
        .map_err(|error| io_error("write package replacement", error))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| io_error("sync package replacement", error))?;
    temporary
        .persist(path)
        .map_err(|error| io_error("replace package", error.error))?;
    sync_directory(parent)
}

fn validate_subassets(subassets: &[EditableSubAsset]) -> Result<(), EditableAssetError> {
    if subassets.len() > MAX_SUBASSETS {
        return Err(EditableAssetError::FieldTooLarge {
            field: "subassets",
            limit: MAX_SUBASSETS,
            actual: subassets.len(),
        });
    }
    let mut names = BTreeSet::new();
    for pair in subassets.windows(2) {
        if pair[0].id >= pair[1].id {
            return Err(EditableAssetError::NonCanonicalSubassets);
        }
    }
    for subasset in subassets {
        validate_text(&subasset.name, MAX_SUBASSET_NAME_BYTES, "subasset name")?;
        if !names.insert(&subasset.name) {
            return Err(EditableAssetError::DuplicateSubAssetName);
        }
    }
    Ok(())
}

fn validate_redirects(
    previous_by_id: &BTreeMap<u64, &EditableSubAsset>,
    next_by_id: &BTreeMap<u64, &EditableSubAsset>,
    redirects: &[EditableSubAssetRedirect],
) -> Result<BTreeMap<u64, u64>, EditableAssetError> {
    let mut targets = BTreeSet::new();
    for pair in redirects.windows(2) {
        if pair[0].from >= pair[1].from {
            return Err(EditableAssetError::NonCanonicalSubAssetRedirects);
        }
    }
    let mut result = BTreeMap::new();
    for redirect in redirects {
        let previous = previous_by_id
            .get(&redirect.from)
            .ok_or(EditableAssetError::UnknownRedirectSource { id: redirect.from })?;
        let next = next_by_id
            .get(&redirect.to)
            .ok_or(EditableAssetError::UnknownRedirectTarget { id: redirect.to })?;
        if next_by_id.contains_key(&redirect.from) {
            return Err(EditableAssetError::RedirectSourceStillPresent { id: redirect.from });
        }
        if previous_by_id.contains_key(&redirect.to) {
            return Err(EditableAssetError::RedirectTargetIsNotNew { id: redirect.to });
        }
        if previous.kind != next.kind {
            return Err(EditableAssetError::RedirectKindMismatch {
                from: redirect.from,
                to: redirect.to,
            });
        }
        if !targets.insert(redirect.to) {
            return Err(EditableAssetError::NonCanonicalSubAssetRedirects);
        }
        result.insert(redirect.from, redirect.to);
    }
    Ok(result)
}

fn validate_editable_path(path: &Path) -> Result<(), EditableAssetError> {
    if path
        .extension()
        .is_none_or(|extension| extension != "prism")
    {
        return Err(EditableAssetError::InvalidFilePath);
    }
    Ok(())
}

fn ensure_regular_file(path: &Path) -> Result<(), EditableAssetError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| io_error("inspect", error))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(EditableAssetError::NonRegularFile);
    }
    Ok(())
}

fn ensure_directory(path: &Path) -> Result<(), EditableAssetError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| io_error("inspect parent", error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(EditableAssetError::NonRegularFile);
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), EditableAssetError> {
    #[cfg(unix)]
    {
        File::open(path)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| io_error("sync directory", error))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn io_error(operation: &'static str, error: io::Error) -> EditableAssetError {
    EditableAssetError::Io {
        operation,
        kind: error.kind(),
    }
}

fn validate_text(value: &str, limit: usize, field: &'static str) -> Result<(), EditableAssetError> {
    if value.is_empty() || value.as_bytes().contains(&0) {
        return Err(EditableAssetError::InvalidText(field));
    }
    if value.len() > limit {
        return Err(EditableAssetError::FieldTooLarge {
            field,
            limit,
            actual: value.len(),
        });
    }
    Ok(())
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_u128(out: &mut Vec<u8>, value: u128) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u32(out, u32::try_from(bytes.len()).expect("validated length"));
    out.extend_from_slice(bytes);
}
fn put_string(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8], EditableAssetError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(EditableAssetError::Truncated)?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or(EditableAssetError::Truncated)?;
        self.offset = end;
        Ok(result)
    }
    fn byte(&mut self) -> Result<u8, EditableAssetError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, EditableAssetError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64, EditableAssetError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn u128(&mut self) -> Result<u128, EditableAssetError> {
        Ok(u128::from_le_bytes(self.array()?))
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], EditableAssetError> {
        self.take(N)?
            .try_into()
            .map_err(|_| EditableAssetError::Truncated)
    }
    fn bytes(&mut self, limit: usize, field: &'static str) -> Result<Vec<u8>, EditableAssetError> {
        let length = usize::try_from(self.u32()?).map_err(|_| EditableAssetError::Truncated)?;
        if length > limit {
            return Err(EditableAssetError::FieldTooLarge {
                field,
                limit,
                actual: length,
            });
        }
        Ok(self.take(length)?.to_vec())
    }
    fn string(&mut self, limit: usize, field: &'static str) -> Result<String, EditableAssetError> {
        let bytes = self.bytes(limit, field)?;
        let value = String::from_utf8(bytes).map_err(|_| EditableAssetError::InvalidText(field))?;
        validate_text(&value, limit, field)?;
        Ok(value)
    }
    fn count(&mut self, limit: usize, field: &'static str) -> Result<usize, EditableAssetError> {
        let count = usize::try_from(self.u32()?).map_err(|_| EditableAssetError::Truncated)?;
        if count > limit {
            return Err(EditableAssetError::FieldTooLarge {
                field,
                limit,
                actual: count,
            });
        }
        Ok(count)
    }
    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_test_asset() -> EditableAsset {
        EditableAsset::new(
            AssetId(1),
            1,
            0,
            Vec::new(),
            SourceRecord::embedded("source", "test/raw", b"source".to_vec()).unwrap(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            b"old".to_vec(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn validated_encoded_len_matches_every_optional_field_combination() {
        for embedded in [false, true] {
            for preview in [None, Some(Vec::new()), Some(b"preview".to_vec())] {
                let mut asset = scan_test_asset();
                asset.canonical_settings = b"settings".to_vec();
                asset.source.uri = "nested/é.source".into();
                asset.source.format = "test/unicode".into();
                if !embedded {
                    asset.source.embedded_source = None;
                }
                asset.preview = preview;
                asset.subassets = vec![EditableSubAsset {
                    id: 7,
                    name: "子对象".into(),
                    kind: 11,
                }];
                asset.dependencies = vec![EditableDependency {
                    asset: AssetId(2),
                    kind: EditableDependencyKind::Soft,
                }];
                asset.subasset_references = vec![EditableSubAssetReference {
                    asset: AssetId(2),
                    subasset: 3,
                    kind: EditableDependencyKind::EditorOnly,
                }];
                let bytes = asset.encode().unwrap();
                assert_eq!(asset.validated_encoded_len().unwrap(), bytes.len());
                assert_eq!(EditableAsset::decode(&bytes).unwrap(), asset);
            }
        }
        let mut asset = scan_test_asset();
        asset.source.embedded_source = Some(Vec::new());
        asset.source.content_hash = ContentHash::digest(&[]);
        asset.authoring_payload.clear();
        assert_eq!(
            asset.validated_encoded_len().unwrap(),
            asset.encode().unwrap().len()
        );
    }

    #[test]
    fn validation_and_constructor_enforce_the_total_encoded_size_before_encoding() {
        let mut asset = scan_test_asset();
        asset.source =
            SourceRecord::embedded("source", "test/raw", vec![0; MAX_EMBEDDED_SOURCE_BYTES])
                .unwrap();
        asset.authoring_payload.clear();
        let remaining = MAX_EDITABLE_ASSET_BYTES - asset.validated_encoded_len().unwrap();
        assert!(remaining < MAX_AUTHORING_PAYLOAD_BYTES);
        asset.authoring_payload = vec![0; remaining];
        assert_eq!(asset.validated_encoded_len(), Ok(MAX_EDITABLE_ASSET_BYTES));
        asset.authoring_payload.push(0);
        let expected = EditableAssetError::PackageTooLarge {
            actual: MAX_EDITABLE_ASSET_BYTES + 1,
        };
        assert_eq!(asset.validate(), Err(expected.clone()));
        assert_eq!(asset.validated_encoded_len(), Err(expected.clone()));
        assert_eq!(asset.encode(), Err(expected.clone()));
        assert_eq!(
            EditableAsset::new(
                asset.asset,
                asset.kind,
                asset.flags,
                asset.canonical_settings,
                asset.source,
                asset.subassets,
                asset.dependencies,
                asset.subasset_references,
                asset.authoring_payload,
                asset.preview,
            ),
            Err(expected)
        );
    }

    #[test]
    fn scan_limits_bound_package_paths_before_decoding() {
        let directory = tempfile::tempdir().unwrap();
        for name in ["a.prism", "b.prism", "c.prism"] {
            fs::write(directory.path().join(name), b"not a package").unwrap();
        }
        let mut limits = WorkspaceScanLimits::new(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES);
        limits.packages = 2;
        assert!(matches!(
            scan_workspace_packages(directory.path(), limits),
            Err(EditableAssetError::WorkspacePackageLimitExceeded { limit: 2 })
        ));
        limits.packages = 3;
        assert_eq!(
            scan_workspace_directory(directory.path(), limits)
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn scan_limits_count_unrelated_entries_and_directories() {
        let files = tempfile::tempdir().unwrap();
        for name in ["a.txt", "b.txt", "c.txt"] {
            fs::write(files.path().join(name), []).unwrap();
        }
        let mut limits = WorkspaceScanLimits::new(1);
        limits.entries = 2;
        assert!(matches!(
            scan_workspace_packages(files.path(), limits),
            Err(EditableAssetError::WorkspaceEntryLimitExceeded { limit: 2 })
        ));
        limits.entries = 3;
        assert!(scan_workspace_packages(files.path(), limits)
            .unwrap()
            .is_empty());

        let directories = tempfile::tempdir().unwrap();
        for name in ["a", "b"] {
            fs::create_dir(directories.path().join(name)).unwrap();
        }
        limits.directories = 2;
        assert!(matches!(
            scan_workspace_packages(directories.path(), limits),
            Err(EditableAssetError::WorkspaceDirectoryLimitExceeded { limit: 2 })
        ));
        limits.directories = 3;
        assert!(scan_workspace_packages(directories.path(), limits)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn package_growth_after_preflight_is_rejected_before_decoding() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("asset.prism");
        scan_test_asset().write_to_path(&path).unwrap();
        let lengths =
            scan_workspace_directory(directory.path(), WorkspaceScanLimits::new(4096)).unwrap();
        let mut replacement = scan_test_asset();
        replacement.authoring_payload.extend_from_slice(b"grown");
        replacement.write_to_path(&path).unwrap();
        assert_eq!(
            read_editable_package_bytes_at_length(&path, Some(lengths[0].1)),
            Err(EditableAssetError::TransactionConflict)
        );
    }

    #[test]
    fn snapshot_recovers_a_transaction_started_after_the_workspace_was_opened() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("asset.prism");
        let original = scan_test_asset();
        original.write_to_path(&path).unwrap();
        let workspace = EditableWorkspace::open(directory.path()).unwrap();
        let mut replacement = original.clone();
        replacement.authoring_payload = b"new payload".to_vec();
        let replacement_bytes = replacement.encode().unwrap();
        let transaction_root = ensure_workspace_transactions_root(directory.path()).unwrap();
        let transaction = create_workspace_transaction(
            &transaction_root,
            &[WorkspaceTransactionRecord {
                relative_path: PathBuf::from("asset.prism"),
                old_bytes: original.encode().unwrap(),
                new_bytes: replacement_bytes.clone(),
            }],
        )
        .unwrap();
        replace_file_bytes(&path, &replacement_bytes).unwrap();
        let captured = workspace.snapshot(4096).unwrap();
        assert_eq!(captured.get(&original.asset), Some(&original));
        assert_eq!(EditableAsset::read_from_path(&path).unwrap(), original);
        assert!(!transaction.exists());
    }

    #[test]
    fn opening_a_workspace_restores_an_uncommitted_package_transaction() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("asset.prism");
        let original = EditableAsset::new(
            AssetId(1),
            1,
            0,
            Vec::new(),
            SourceRecord::embedded("source", "test/raw", b"source".to_vec()).unwrap(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            b"old".to_vec(),
            None,
        )
        .unwrap();
        original.write_to_path(&path).unwrap();
        let old_bytes = original.encode().unwrap();
        let mut replacement = original.clone();
        replacement.authoring_payload = b"new".to_vec();
        let new_bytes = replacement.encode().unwrap();

        let transaction_root = ensure_workspace_transactions_root(directory.path()).unwrap();
        let transaction = create_workspace_transaction(
            &transaction_root,
            &[WorkspaceTransactionRecord {
                relative_path: PathBuf::from("asset.prism"),
                old_bytes: old_bytes.clone(),
                new_bytes: new_bytes.clone(),
            }],
        )
        .unwrap();
        replace_file_bytes(&path, &new_bytes).unwrap();

        EditableWorkspace::open(directory.path()).unwrap();
        assert_eq!(fs::read(&path).unwrap(), old_bytes);
        assert!(!transaction.exists());
    }

    #[test]
    fn opening_a_workspace_keeps_a_committed_package_transaction() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("asset.prism");
        let original = EditableAsset::new(
            AssetId(1),
            1,
            0,
            Vec::new(),
            SourceRecord::embedded("source", "test/raw", b"source".to_vec()).unwrap(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            b"old".to_vec(),
            None,
        )
        .unwrap();
        original.write_to_path(&path).unwrap();
        let old_bytes = original.encode().unwrap();
        let mut replacement = original.clone();
        replacement.authoring_payload = b"new".to_vec();
        let new_bytes = replacement.encode().unwrap();

        let transaction_root = ensure_workspace_transactions_root(directory.path()).unwrap();
        let transaction = create_workspace_transaction(
            &transaction_root,
            &[WorkspaceTransactionRecord {
                relative_path: PathBuf::from("asset.prism"),
                old_bytes,
                new_bytes: new_bytes.clone(),
            }],
        )
        .unwrap();
        replace_file_bytes(&path, &new_bytes).unwrap();
        write_new_file(
            &transaction.join("committed"),
            EDITABLE_TRANSACTION_COMMITTED,
        )
        .unwrap();

        EditableWorkspace::open(directory.path()).unwrap();
        assert_eq!(fs::read(&path).unwrap(), new_bytes);
        assert!(!transaction.exists());
    }

    #[test]
    fn opening_a_workspace_discards_an_incomplete_transaction_stage() {
        let directory = tempfile::tempdir().unwrap();
        let transaction_root = ensure_workspace_transactions_root(directory.path()).unwrap();
        let incomplete = transaction_root.join("incomplete");
        fs::create_dir(&incomplete).unwrap();
        write_new_file(&incomplete.join("00000000.old"), b"partial").unwrap();

        EditableWorkspace::open(directory.path()).unwrap();
        assert!(!incomplete.exists());
    }
}
