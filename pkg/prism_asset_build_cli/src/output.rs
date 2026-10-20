//! CLI event protocol v1. Each JSON invocation emits `command_started` (sequence
//! 0), followed by exactly one `command_succeeded` or `command_failed` (sequence
//! 1). Every envelope has the same fields, including explicit nulls. Exit codes
//! are 0 for success, 1 for operation/output errors, and 2 for invalid arguments.
//! Unknown/unparsed commands use a null command. Help is a successful result.
//!
//! Result shape is selected by the command, not by parsing human output. Asset
//! IDs and kinds are decimal strings (u128); keys/digests are lowercase hex.
//! Arrays preserve canonical core order. No argv, timestamps, key bytes or raw
//! signatures are emitted. Output write failure cannot guarantee a final event;
//! it exits nonzero and never retries an operation that may have committed.

use prism_asset_build_core::{
    ArtifactDescriptor, AssetId, ContentHash, InvalidationReason, ProfileFieldChange,
    ValidationReport,
};
use prism_asset_container::RuntimeManifest;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum CommandResult {
    Plan(PlanResult),
    Publish(PublishResult),
    Package(PackageResult),
    Validate(PackageResult),
    Patch(PatchResult),
    Inspect(InspectResult),
    Why(RebuildExplanationResult),
    Help { help: String },
}

#[derive(Debug, Serialize)]
pub struct PlanResult {
    pub source_path: String,
    pub source_artifact_key: String,
    pub image_artifact_key: String,
    pub image_asset_id: String,
}

#[derive(Debug, Serialize)]
pub struct PublishResult {
    pub revision: u64,
    pub build_manifest_hash: String,
    pub source_artifact_key: String,
    pub image_artifact_key: String,
    pub image_asset_id: String,
    pub image_content_hash: String,
    pub image_width: u32,
    pub image_height: u32,
    pub image_mip_levels: usize,
    pub cache_hits: usize,
    pub determinism_verified_jobs: Option<usize>,
    pub artifacts: Vec<ArtifactResult>,
}

#[derive(Debug, Serialize)]
pub struct ArtifactResult {
    pub asset_id: String,
    pub artifact_key: String,
    pub kind: String,
    pub schema: u32,
    pub bytes: u64,
    pub content_hash: String,
    pub dependencies: Vec<String>,
}

impl ArtifactResult {
    pub fn new(asset: AssetId, descriptor: &ArtifactDescriptor) -> Self {
        Self {
            asset_id: asset.0.to_string(),
            artifact_key: descriptor.key.to_hex(),
            kind: descriptor.kind.to_string(),
            schema: descriptor.schema,
            bytes: descriptor.byte_len,
            content_hash: descriptor.content_hash.to_hex(),
            dependencies: descriptor
                .dependencies
                .iter()
                .map(|key| key.to_hex())
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InspectResult {
    pub current_revision: Option<u64>,
    pub build_manifest_hash: Option<String>,
    pub asset_count: usize,
    pub artifacts: Vec<ArtifactResult>,
}

#[derive(Debug, Serialize)]
pub struct RebuildExplanationResult {
    pub revision: u64,
    pub direct_jobs: Vec<String>,
    pub affected_jobs: Vec<String>,
    pub changes: Vec<RebuildChangeResult>,
}

#[derive(Debug, Serialize)]
pub struct RebuildChangeResult {
    pub job_id: String,
    pub kind: &'static str,
    pub before_path: Option<String>,
    pub before_hash: Option<String>,
    pub after_path: Option<String>,
    pub after_hash: Option<String>,
    pub profile_fields: Vec<ProfileFieldChangeResult>,
}

impl From<&InvalidationReason> for RebuildChangeResult {
    fn from(reason: &InvalidationReason) -> Self {
        let mut result = Self {
            job_id: reason.job().0.to_string(),
            kind: "source",
            before_path: None,
            before_hash: None,
            after_path: None,
            after_hash: None,
            profile_fields: Vec::new(),
        };
        match reason {
            InvalidationReason::Source { before, after, .. } => {
                result.before_path = before
                    .as_ref()
                    .map(|identity| identity.logical_path.clone());
                result.before_hash = before
                    .as_ref()
                    .map(|identity| identity.content_hash.to_hex());
                result.after_path = after.as_ref().map(|identity| identity.logical_path.clone());
                result.after_hash = after
                    .as_ref()
                    .map(|identity| identity.content_hash.to_hex());
            }
            InvalidationReason::Settings { before, after, .. } => {
                result.kind = "settings";
                result.before_hash = before.map(ContentHash::to_hex);
                result.after_hash = after.map(ContentHash::to_hex);
            }
            InvalidationReason::Profile {
                before,
                after,
                fields,
                ..
            } => {
                result.kind = "profile";
                result.before_hash = before.map(hex_bytes);
                result.after_hash = after.map(hex_bytes);
                result.profile_fields = fields.iter().map(ProfileFieldChangeResult::from).collect();
            }
        }
        result
    }
}

#[derive(Debug, Serialize)]
pub struct ProfileFieldChangeResult {
    pub field: String,
    pub before_hash: Option<String>,
    pub after_hash: Option<String>,
}

impl From<&ProfileFieldChange> for ProfileFieldChangeResult {
    fn from(change: &ProfileFieldChange) -> Self {
        Self {
            field: change.field.clone(),
            before_hash: change.before_hash.map(hex_bytes),
            after_hash: change.after_hash.map(hex_bytes),
        }
    }
}

fn hex_bytes(bytes: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Debug, Serialize)]
pub struct QualityResult {
    pub checked_assets: usize,
    pub checked_artifacts: usize,
    pub checked_blobs: usize,
    pub checked_bytes: u64,
    pub checked_dependency_edges: usize,
}

impl From<ValidationReport> for QualityResult {
    fn from(report: ValidationReport) -> Self {
        Self {
            checked_assets: report.checked_assets,
            checked_artifacts: report.checked_artifacts,
            checked_blobs: report.checked_blobs,
            checked_bytes: report.checked_bytes,
            checked_dependency_edges: report.checked_dependency_edges,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SignatureStatus {
    Unsigned,
    /// This invocation wrote the signature; verification requires a trust input.
    Signed,
    /// A signature file preexisted, but this invocation did not verify it.
    PresentUnverified,
    VerifiedPublicKey,
    VerifiedTrustPolicy,
}

#[derive(Debug, Serialize)]
pub struct PackageResult {
    pub revision: u64,
    pub output_directory: String,
    pub build_manifest_hash: String,
    pub runtime_manifest_hash: String,
    pub runtime_manifest_bytes: usize,
    pub container_count: usize,
    pub containers: Vec<ContainerResult>,
    pub files: Vec<FileResult>,
    pub quality: QualityResult,
    pub signature: SignatureStatus,
}

#[derive(Debug, Serialize)]
pub struct ContainerResult {
    pub name: String,
    pub bytes: usize,
    pub content_hash: String,
    pub entries: usize,
}

/// Uses the generated routing manifest instead of cloning/mounting each entire
/// container merely to produce CLI statistics. GUID aliases count once.
pub fn container_results(
    containers: &BTreeMap<String, Vec<u8>>,
    manifest: &RuntimeManifest,
) -> Vec<ContainerResult> {
    let mut entries: BTreeMap<&str, BTreeSet<_>> = BTreeMap::new();
    for route in manifest.routes() {
        entries
            .entry(&route.container_name)
            .or_default()
            .insert(route.container_guid);
    }
    containers
        .iter()
        .map(|(name, bytes)| ContainerResult {
            name: name.clone(),
            bytes: bytes.len(),
            content_hash: ContentHash::digest(bytes).to_hex(),
            entries: entries.get(name.as_str()).map_or(0, BTreeSet::len),
        })
        .collect()
}

#[derive(Debug, Serialize)]
pub struct FileResult {
    /// Relative to the result's output directory. Signature material is omitted.
    pub path: String,
    pub bytes: usize,
    pub content_hash: String,
}

impl FileResult {
    pub fn new(path: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            path: path.into(),
            bytes: bytes.len(),
            content_hash: ContentHash::digest(bytes).to_hex(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct PatchResult {
    pub base_revision: u64,
    pub target_revision: u64,
    pub output_directory: String,
    pub removed_routes: usize,
    pub patch_container_count: usize,
    pub patch_manifest_hash: String,
    pub target_manifest_hash: String,
    pub containers: Vec<ContainerResult>,
    pub files: Vec<FileResult>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum EventKind {
    CommandStarted,
    CommandSucceeded,
    CommandFailed,
}

#[derive(Serialize)]
struct ErrorResult<'a> {
    code: &'a str,
    message: &'a str,
}

#[derive(Serialize)]
struct Event<'a> {
    schema_version: u32,
    sequence: u32,
    event: EventKind,
    command: Option<&'a str>,
    result: Option<&'a CommandResult>,
    error: Option<ErrorResult<'a>>,
    exit_code: Option<u8>,
}

/// Owns stdout for the invocation. Command implementations never print.
pub struct Output<W: Write> {
    writer: W,
    json: bool,
}

impl<W: Write> Output<W> {
    pub fn new(writer: W, json: bool) -> Self {
        Self { writer, json }
    }

    pub fn started(&mut self, command: Option<&str>) -> io::Result<()> {
        if self.json {
            self.event(Event {
                schema_version: 1,
                sequence: 0,
                event: EventKind::CommandStarted,
                command,
                result: None,
                error: None,
                exit_code: None,
            })?;
        }
        Ok(())
    }

    pub fn succeeded(&mut self, command: Option<&str>, result: &CommandResult) -> io::Result<()> {
        if self.json {
            self.event(Event {
                schema_version: 1,
                sequence: 1,
                event: EventKind::CommandSucceeded,
                command,
                result: Some(result),
                error: None,
                exit_code: Some(0),
            })
        } else {
            result.write_human(&mut self.writer)?;
            self.writer.flush()
        }
    }

    pub fn failed(
        &mut self,
        command: Option<&str>,
        code: &str,
        message: &str,
        exit_code: u8,
    ) -> io::Result<()> {
        if self.json {
            self.event(Event {
                schema_version: 1,
                sequence: 1,
                event: EventKind::CommandFailed,
                command,
                result: None,
                error: Some(ErrorResult { code, message }),
                exit_code: Some(exit_code),
            })
        } else {
            writeln!(io::stderr().lock(), "prism asset build failed: {message}")
        }
    }

    fn event(&mut self, event: Event<'_>) -> io::Result<()> {
        // Serialize completely before touching stdout; valid escaped JSON is
        // never interleaved with human diagnostics or split across flushes.
        let mut line = serde_json::to_vec(&event).map_err(io::Error::other)?;
        line.push(b'\n');
        self.writer.write_all(&line)?;
        self.writer.flush()
    }
}

// Quote user-controlled text so a path or profile field cannot add human-output
// records through whitespace, newlines, or delimiter characters.
fn text_atom(value: &str) -> String {
    format!("{value:?}")
}

impl CommandResult {
    fn write_human(&self, out: &mut impl Write) -> io::Result<()> {
        match self {
            Self::Plan(result) => {
                writeln!(out, "source_path={}", result.source_path)?;
                writeln!(out, "source_artifact_key={}", result.source_artifact_key)?;
                writeln!(out, "image_artifact_key={}", result.image_artifact_key)?;
                writeln!(out, "image_asset_id={}", result.image_asset_id)?;
            }
            Self::Publish(result) => {
                writeln!(out, "revision={}", result.revision)?;
                writeln!(out, "source_artifact_key={}", result.source_artifact_key)?;
                writeln!(out, "image_artifact_key={}", result.image_artifact_key)?;
                writeln!(out, "image_content_hash={}", result.image_content_hash)?;
                writeln!(
                    out,
                    "image_dimensions={}x{}",
                    result.image_width, result.image_height
                )?;
                writeln!(out, "image_mip_levels={}", result.image_mip_levels)?;
                writeln!(out, "cache_hits={}", result.cache_hits)?;
                if let Some(count) = result.determinism_verified_jobs {
                    writeln!(out, "determinism_verified_jobs={count}")?;
                }
            }
            Self::Inspect(result) => {
                let Some(revision) = result.current_revision else {
                    return writeln!(out, "current_revision=none");
                };
                writeln!(out, "current_revision={revision}")?;
                writeln!(out, "asset_count={}", result.asset_count)?;
                for asset in &result.artifacts {
                    writeln!(
                        out,
                        "asset={} key={} kind={} schema={} bytes={} hash={} dependencies={}",
                        asset.asset_id,
                        asset.artifact_key,
                        asset.kind,
                        asset.schema,
                        asset.bytes,
                        asset.content_hash,
                        asset.dependencies.len()
                    )?;
                }
            }
            Self::Why(result) => {
                writeln!(out, "revision={}", result.revision)?;
                writeln!(out, "direct_jobs={}", result.direct_jobs.join(","))?;
                writeln!(out, "affected_jobs={}", result.affected_jobs.join(","))?;
                for change in &result.changes {
                    writeln!(
                        out,
                        "change job={} kind={} before_path={} before_hash={} after_path={} after_hash={}",
                        change.job_id,
                        change.kind,
                        change.before_path.as_deref().map(text_atom).unwrap_or_else(|| "none".into()),
                        change.before_hash.as_deref().unwrap_or("none"),
                        change.after_path.as_deref().map(text_atom).unwrap_or_else(|| "none".into()),
                        change.after_hash.as_deref().unwrap_or("none"),
                    )?;
                    for field in &change.profile_fields {
                        writeln!(
                            out,
                            "profile_field job={} field={} before_hash={} after_hash={}",
                            change.job_id,
                            text_atom(&field.field),
                            field.before_hash.as_deref().unwrap_or("none"),
                            field.after_hash.as_deref().unwrap_or("none"),
                        )?;
                    }
                }
            }
            Self::Package(result) | Self::Validate(result) => {
                let quality = &result.quality;
                writeln!(out, "quality_checked_assets={}", quality.checked_assets)?;
                writeln!(
                    out,
                    "quality_checked_artifacts={}",
                    quality.checked_artifacts
                )?;
                writeln!(out, "quality_checked_blobs={}", quality.checked_blobs)?;
                writeln!(out, "quality_checked_bytes={}", quality.checked_bytes)?;
                writeln!(
                    out,
                    "quality_checked_dependency_edges={}",
                    quality.checked_dependency_edges
                )?;
                let packaged = matches!(self, Self::Package(_));
                let label = if packaged {
                    "packaged_revision"
                } else {
                    "validated_revision"
                };
                writeln!(out, "{label}={}", result.revision)?;
                writeln!(out, "build_manifest_hash={}", result.build_manifest_hash)?;
                writeln!(
                    out,
                    "runtime_manifest_hash={}",
                    result.runtime_manifest_hash
                )?;
                if packaged {
                    writeln!(
                        out,
                        "runtime_manifest_bytes={}",
                        result.runtime_manifest_bytes
                    )?;
                }
                writeln!(out, "container_count={}", result.container_count)?;
                if packaged {
                    for container in &result.containers {
                        writeln!(
                            out,
                            "container={} bytes={} hash={} entries={}",
                            container.name,
                            container.bytes,
                            container.content_hash,
                            container.entries
                        )?;
                    }
                }
            }
            Self::Patch(result) => {
                writeln!(out, "base_revision={}", result.base_revision)?;
                writeln!(out, "target_revision={}", result.target_revision)?;
                writeln!(out, "removed_routes={}", result.removed_routes)?;
                writeln!(
                    out,
                    "patch_container_count={}",
                    result.patch_container_count
                )?;
                writeln!(out, "patch_manifest_hash={}", result.patch_manifest_hash)?;
                writeln!(out, "target_manifest_hash={}", result.target_manifest_hash)?;
            }
            Self::Help { help } => writeln!(out, "{help}")?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_why_keeps_paths_and_profile_fields_on_their_own_record() {
        let mut output = Output::new(Vec::new(), false);
        let hostile = "textures/雪\"\nrevision=999\t.png";
        output
            .succeeded(
                Some("why"),
                &CommandResult::Why(RebuildExplanationResult {
                    revision: 7,
                    direct_jobs: vec!["1".into()],
                    affected_jobs: vec!["1".into()],
                    changes: vec![RebuildChangeResult {
                        job_id: "1".into(),
                        kind: "profile",
                        before_path: None,
                        before_hash: None,
                        after_path: Some(hostile.into()),
                        after_hash: None,
                        profile_fields: vec![ProfileFieldChangeResult {
                            field: hostile.into(),
                            before_hash: None,
                            after_hash: None,
                        }],
                    }],
                }),
            )
            .unwrap();
        let rendered = String::from_utf8(output.writer).unwrap();
        assert_eq!(rendered.lines().count(), 5);
        assert!(!rendered.lines().any(|line| line == "revision=999"));
        assert!(rendered.contains("before_path=none before_hash=none"));
        assert!(rendered.contains(&format!("after_path={hostile:?}")));
        assert!(rendered.contains(&format!("field={hostile:?}")));
    }

    #[test]
    fn version_one_envelope_and_escaping_are_stable() {
        let mut output = Output::new(Vec::new(), true);
        output.started(None).unwrap();
        let message = "quote=\" slash=\\ newline=\n tab=\t nul=\0 unicode=雪";
        output
            .failed(None, "invalid_arguments", message, 2)
            .unwrap();
        let text = String::from_utf8(output.writer).unwrap();
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0],
            r#"{"schema_version":1,"sequence":0,"event":"command_started","command":null,"result":null,"error":null,"exit_code":null}"#
        );
        let end: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(end["error"]["message"], message);
        assert_eq!(end["exit_code"], 2);
    }

    #[test]
    fn human_mode_does_not_print_start_events() {
        let mut output = Output::new(Vec::new(), false);
        output.started(Some("inspect")).unwrap();
        assert!(output.writer.is_empty());
        output
            .succeeded(
                Some("inspect"),
                &CommandResult::Inspect(InspectResult {
                    current_revision: None,
                    build_manifest_hash: None,
                    asset_count: 0,
                    artifacts: Vec::new(),
                }),
            )
            .unwrap();
        assert_eq!(output.writer, b"current_revision=none\n");
    }

    struct BrokenWriter;

    impl Write for BrokenWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::BrokenPipe.into())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn broken_stdout_propagates_instead_of_claiming_success() {
        let mut output = Output::new(BrokenWriter, true);
        assert_eq!(
            output.started(Some("plan")).unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        assert_eq!(
            output
                .failed(Some("plan"), "io_error", "failure", 1)
                .unwrap_err()
                .kind(),
            io::ErrorKind::BrokenPipe
        );
        assert_eq!(
            output
                .succeeded(
                    Some("plan"),
                    &CommandResult::Help {
                        help: "help".into()
                    }
                )
                .unwrap_err()
                .kind(),
            io::ErrorKind::BrokenPipe
        );
    }

    struct FlushFailure;

    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }

    #[test]
    fn event_flush_failure_propagates_to_exit_status() {
        let mut output = Output::new(FlushFailure, true);
        assert!(output.started(Some("plan")).is_err());
    }
}
