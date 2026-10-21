//! Saved authoring payload CLI bridge. Source records are never opened here.

use crate::args::{EditableBuild, EditablePlan};
use crate::error::CommandError;
use crate::output::ArtifactResult;
use prism_asset_build_core::{
    AssetId, BuildResources, BuildRevision, ContentHash, DerivedDataCache, EditableBuildOptions,
    EditableBuildPlan, EditableCookerRegistry, EditableWorkspace, RevisionStore,
};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::Path;

/// Shared selection for planning, cooking and reconstruction from a committed
/// manifest. Packaging uses the same fingerprints and snapshot rules as cooking.
pub struct WorkspaceInput<'a> {
    pub workspace: Option<&'a Path>,
    pub roots: &'a [u128],
    pub max_snapshot_bytes: Option<u64>,
    pub max_materialized_payload_bytes: Option<u64>,
    pub profile_hash: Option<&'a str>,
    pub toolchain_fingerprint: Option<&'a str>,
    pub environment_fingerprint: Option<&'a str>,
}

impl WorkspaceInput<'_> {
    pub fn capture(&self) -> Result<EditableBuildPlan, CommandError> {
        let path = self.workspace.ok_or_else(|| {
            CommandError::InvalidInput("editable selection requires --workspace".into())
        })?;
        if self.roots.is_empty() || self.roots.len() > 65_536 {
            return Err(CommandError::InvalidInput(
                "supply between 1 and 65536 --root arguments".into(),
            ));
        }
        let mut options = EditableBuildOptions::default();
        if let Some(limit) = self.max_snapshot_bytes {
            options.max_snapshot_bytes = limit;
        }
        if let Some(limit) = self.max_materialized_payload_bytes {
            options.max_materialized_payload_bytes = limit;
        }
        if let Some(hash) = self.profile_hash {
            options.profile_hash = profile_hash(hash)?;
        }
        options.toolchain_fingerprint = fingerprint(self.toolchain_fingerprint)?;
        options.environment_fingerprint = fingerprint(self.environment_fingerprint)?;
        let roots = self
            .roots
            .iter()
            .copied()
            .map(AssetId)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        Ok(EditableBuildPlan::from_workspace(
            &EditableWorkspace::open(path)?,
            &roots,
            &EditableCookerRegistry::with_builtin(),
            &options,
        )?)
    }

    /// Reject mixed identity modes before opening the workspace or store.
    pub fn package_plan(
        &self,
        mapping: Option<&Path>,
        asset_id: Option<u128>,
        logical_path: Option<&str>,
    ) -> Result<Option<EditableBuildPlan>, CommandError> {
        if self.workspace.is_some() {
            if mapping.is_some() || asset_id.is_some() || logical_path.is_some() {
                return Err(CommandError::InvalidInput(
                    "--workspace cannot be combined with --mapping, --asset-id or --logical-path"
                        .into(),
                ));
            }
            return self.capture().map(Some);
        }
        if !self.roots.is_empty()
            || self.max_snapshot_bytes.is_some()
            || self.max_materialized_payload_bytes.is_some()
            || self.profile_hash.is_some()
            || self.toolchain_fingerprint.is_some()
            || self.environment_fingerprint.is_some()
        {
            return Err(CommandError::InvalidInput(
                "editable roots, budgets and fingerprints require --workspace".into(),
            ));
        }
        Ok(None)
    }
}

fn fingerprint(value: Option<&str>) -> Result<Vec<u8>, CommandError> {
    let value = value.unwrap_or_default();
    if value.len() > 1024 * 1024 {
        return Err(CommandError::InvalidInput(
            "editable fingerprint exceeds 1 MiB".into(),
        ));
    }
    Ok(value.as_bytes().to_vec())
}

fn profile_hash(value: &str) -> Result<[u8; 32], CommandError> {
    let invalid = || {
        CommandError::InvalidInput(
            "--profile-hash must contain exactly 64 hexadecimal digits".into(),
        )
    };
    if value.len() != 64 {
        return Err(invalid());
    }
    let mut hash = [0; 32];
    for (out, pair) in hash.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        let high = (pair[0] as char).to_digit(16).ok_or_else(invalid)?;
        let low = (pair[1] as char).to_digit(16).ok_or_else(invalid)?;
        *out = ((high << 4) | low) as u8;
    }
    Ok(hash)
}

#[derive(Debug, Serialize)]
pub struct EditablePlanResult {
    pub roots: Vec<String>,
    pub execution_order: Vec<String>,
    pub assets: Vec<PlannedAsset>,
}

#[derive(Debug, Serialize)]
pub struct PlannedAsset {
    pub asset_id: String,
    pub job_id: String,
    pub processor_id: String,
    pub processor_version: u32,
    pub kind: String,
    pub schema: u32,
    pub alignment: u32,
}

#[derive(Debug, Serialize)]
pub struct EditableBuildResult {
    pub revision: u64,
    pub build_manifest_hash: String,
    pub cache_hits: usize,
    pub determinism_verified_jobs: Option<usize>,
    pub artifacts: Vec<ArtifactResult>,
}

pub fn plan(command: EditablePlan) -> Result<EditablePlanResult, CommandError> {
    let plan = WorkspaceInput {
        workspace: Some(&command.workspace),
        roots: &command.root,
        max_snapshot_bytes: command.max_snapshot_bytes,
        max_materialized_payload_bytes: command.max_materialized_payload_bytes,
        profile_hash: command.profile_hash.as_deref(),
        toolchain_fingerprint: command.toolchain_fingerprint.as_deref(),
        environment_fingerprint: command.environment_fingerprint.as_deref(),
    }
    .capture()?;
    let order = plan
        .graph()
        .execution_order()
        .map_err(|error| CommandError::InvalidInput(error.to_string()))?;
    Ok(EditablePlanResult {
        roots: command
            .root
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|id| id.to_string())
            .collect(),
        execution_order: order.iter().map(|job| job.0.to_string()).collect(),
        assets: plan
            .specs()
            .iter()
            .map(|(job, spec)| PlannedAsset {
                asset_id: spec.asset.0.to_string(),
                job_id: job.0.to_string(),
                processor_id: spec.processor_id.to_string(),
                processor_version: spec.processor_version,
                kind: spec.kind.to_string(),
                schema: spec.schema,
                alignment: spec.alignment,
            })
            .collect(),
    })
}

pub fn build(command: EditableBuild) -> Result<EditableBuildResult, CommandError> {
    if command.max_concurrency == 0 || command.cpu_threads == 0 {
        return Err(CommandError::InvalidInput(
            "--max-concurrency and --cpu-threads must be positive".into(),
        ));
    }
    let plan = WorkspaceInput {
        workspace: Some(&command.workspace),
        roots: &command.root,
        max_snapshot_bytes: command.max_snapshot_bytes,
        max_materialized_payload_bytes: command.max_materialized_payload_bytes,
        profile_hash: command.profile_hash.as_deref(),
        toolchain_fingerprint: command.toolchain_fingerprint.as_deref(),
        environment_fingerprint: command.environment_fingerprint.as_deref(),
    }
    .capture()?;
    let store = RevisionStore::open(&command.store)?;
    let revision = match store.current_revision()? {
        Some(revision) => revision
            .0
            .checked_add(1)
            .ok_or_else(|| CommandError::InvalidInput("revision space is exhausted".into()))?,
        None => 0,
    };
    let mut ddc = DerivedDataCache::open(command.store.join("ddc"))?;
    // Built-in saved-image cookers declare CPU and memory only. Other tokens
    // remain zero; introducing another cooker must explicitly budget its demand.
    let budget = BuildResources {
        cpu_threads: command.cpu_threads,
        memory_bytes: command.memory_bytes,
        ..BuildResources::default()
    };
    let execution = plan.execute(
        &store,
        &mut ddc,
        BuildRevision(revision),
        command.max_concurrency,
        budget,
        command.verify_determinism,
    )?;
    let manifest = &execution.published.manifest;
    Ok(EditableBuildResult {
        revision: manifest.revision.0,
        build_manifest_hash: ContentHash::digest(&manifest.canonical_bytes()).to_hex(),
        cache_hits: execution.cache_hits.len(),
        determinism_verified_jobs: command
            .verify_determinism
            .then_some(execution.artifact_keys.len()),
        artifacts: manifest
            .entries
            .iter()
            .map(|(asset, descriptor)| ArtifactResult::new(*asset, descriptor))
            .collect(),
    })
}
