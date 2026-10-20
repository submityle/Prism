//! Read-only graph and diff presentation over core committed-manifest evidence.

use crate::{args, error::CommandError};
use prism_asset_build_core::{
    plan_patch, ArtifactDescriptor, ArtifactStorageClass, AssetId, BuildRevision, ContentHash,
    Manifest, PatchChange, RevisionStore,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::path::Path;

fn invalid(message: impl Into<String>) -> CommandError {
    CommandError::InvalidInput(message.into())
}

struct Evidence {
    manifest: Manifest,
    hash: ContentHash,
}
fn load(root: &Path, revision: u64) -> Result<Evidence, CommandError> {
    let (manifest, hash) = RevisionStore::read_committed_manifest(root, BuildRevision(revision))?;
    Ok(Evidence { manifest, hash })
}

#[derive(Debug, Serialize)]
pub struct DescriptorResult {
    artifact_key: String,
    kind: String,
    schema: u32,
    storage_class: &'static str,
    content_hash: String,
    bytes: u64,
    alignment: u32,
    dependencies: Vec<String>,
}
impl From<&ArtifactDescriptor> for DescriptorResult {
    fn from(d: &ArtifactDescriptor) -> Self {
        Self {
            artifact_key: d.key.to_hex(),
            kind: d.kind.to_string(),
            schema: d.schema,
            storage_class: match d.storage_class {
                ArtifactStorageClass::Shared => "shared",
                ArtifactStorageClass::DriverLocal => "driver_local",
            },
            content_hash: d.content_hash.to_hex(),
            bytes: d.byte_len,
            alignment: d.alignment,
            dependencies: d.dependencies.iter().map(|key| key.to_hex()).collect(),
        }
    }
}
impl DescriptorResult {
    fn write_human(&self, out: &mut impl Write, prefix: &str) -> io::Result<()> {
        writeln!(out, "{prefix} key={} kind={} schema={} storage_class={} hash={} bytes={} alignment={} dependencies={}",
            self.artifact_key, self.kind, self.schema, self.storage_class, self.content_hash,
            self.bytes, self.alignment, self.dependencies.join(","))
    }
}

#[derive(Debug, Serialize)]
pub struct GraphNode {
    asset_ids: Vec<String>,
    descriptor: DescriptorResult,
}
#[derive(Debug, Serialize)]
pub struct GraphEdge {
    artifact_key: String,
    dependency_key: String,
    dependency_index: usize,
}
#[derive(Debug, Serialize)]
pub struct GraphResult {
    revision: u64,
    build_manifest_hash: String,
    selected_asset_id: Option<String>,
    direction: &'static str,
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
}

pub fn graph(command: args::Graph) -> Result<GraphResult, CommandError> {
    if command.reverse && command.asset_id.is_none() {
        return Err(invalid("--reverse requires --asset-id"));
    }
    let evidence = load(&command.store, command.revision)?;
    let mut descriptors = BTreeMap::new();
    let mut owners = BTreeMap::<_, Vec<_>>::new();
    for (&asset, descriptor) in &evidence.manifest.entries {
        descriptors.insert(descriptor.key, descriptor);
        owners
            .entry(descriptor.key)
            .or_default()
            .push(asset.0.to_string());
    }
    let mut reverse = BTreeMap::<_, Vec<_>>::new();
    if command.reverse {
        for (&key, descriptor) in &descriptors {
            for &dependency in &descriptor.dependencies {
                reverse.entry(dependency).or_default().push(key);
            }
        }
    }
    let mut selected = BTreeSet::new();
    if let Some(asset) = command.asset_id {
        let descriptor = evidence
            .manifest
            .entries
            .get(&AssetId(asset))
            .ok_or_else(|| {
                invalid(format!(
                    "AssetId {asset} is absent from revision {}",
                    command.revision
                ))
            })?;
        let mut pending = vec![descriptor.key];
        selected.insert(descriptor.key);
        while let Some(key) = pending.pop() {
            let adjacent = if command.reverse {
                reverse.get(&key).map(Vec::as_slice).unwrap_or(&[])
            } else {
                &descriptors[&key].dependencies
            };
            for &next in adjacent {
                if selected.insert(next) {
                    pending.push(next);
                }
            }
        }
    } else {
        selected.extend(descriptors.keys().copied());
    }
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for key in &selected {
        let descriptor = descriptors[key];
        nodes.push(GraphNode {
            asset_ids: owners.remove(key).unwrap(),
            descriptor: descriptor.into(),
        });
        for (dependency_index, dependency) in descriptor.dependencies.iter().enumerate() {
            if selected.contains(dependency) {
                edges.push(GraphEdge {
                    artifact_key: key.to_hex(),
                    dependency_key: dependency.to_hex(),
                    dependency_index,
                });
            }
        }
    }
    Ok(GraphResult {
        revision: command.revision,
        build_manifest_hash: evidence.hash.to_hex(),
        selected_asset_id: command.asset_id.map(|asset| asset.to_string()),
        direction: if command.asset_id.is_none() {
            "all"
        } else if command.reverse {
            "dependents"
        } else {
            "dependencies"
        },
        nodes,
        edges,
    })
}
impl GraphResult {
    pub fn write_human(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(out, "revision={} build_manifest_hash={} direction={} selected_asset_id={} nodes={} edges={}", self.revision, self.build_manifest_hash, self.direction, self.selected_asset_id.as_deref().unwrap_or("none"), self.nodes.len(), self.edges.len())?;
        for node in &self.nodes {
            node.descriptor
                .write_human(out, &format!("node assets={}", node.asset_ids.join(",")))?;
        }
        for edge in &self.edges {
            writeln!(
                out,
                "edge artifact_key={} dependency_key={} dependency_index={}",
                edge.artifact_key, edge.dependency_key, edge.dependency_index
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct DiffEntry {
    asset_id: String,
    change: &'static str,
    changed_fields: Vec<&'static str>,
    base: Option<DescriptorResult>,
    target: Option<DescriptorResult>,
    content_reused: bool,
}
#[derive(Debug, Serialize)]
pub struct DiffResult {
    base_revision: u64,
    target_revision: u64,
    base_manifest_hash: String,
    target_manifest_hash: String,
    added: usize,
    removed: usize,
    changed: usize,
    unchanged: usize,
    base_bytes: u64,
    target_bytes: u64,
    download_bytes: u64,
    install_growth_bytes: i128,
    new_content: Vec<String>,
    entries: Vec<DiffEntry>,
}

pub fn diff(command: args::Diff) -> Result<DiffResult, CommandError> {
    let base = load(&command.store, command.base_revision)?;
    let target = load(&command.store, command.target_revision)?;
    let mut lengths = BTreeMap::new();
    for d in base
        .manifest
        .entries
        .values()
        .chain(target.manifest.entries.values())
    {
        if lengths
            .insert(d.content_hash, d.byte_len)
            .is_some_and(|old| old != d.byte_len)
        {
            return Err(invalid(
                "content hash byte lengths differ between revisions",
            ));
        }
    }
    // plan_patch's monotonic-revision precondition is a publication policy,
    // whereas diff also permits equal and reverse comparisons. Only its local
    // comparison operands are relabeled; evidence and output retain real IDs.
    let mut base_operand = base.manifest;
    let mut target_operand = target.manifest;
    base_operand.revision = BuildRevision(0);
    target_operand.revision = BuildRevision(1);
    let plan =
        plan_patch(&base_operand, &target_operand).map_err(|error| invalid(error.to_string()))?;
    let total = |manifest: &Manifest| {
        manifest.entries.values().try_fold(0u64, |sum, d| {
            sum.checked_add(d.byte_len)
                .ok_or_else(|| invalid("descriptor byte total overflows u64"))
        })
    };
    let mut result = DiffResult {
        base_revision: command.base_revision,
        target_revision: command.target_revision,
        base_manifest_hash: base.hash.to_hex(),
        target_manifest_hash: target.hash.to_hex(),
        added: 0,
        removed: 0,
        changed: 0,
        unchanged: 0,
        base_bytes: total(&base_operand)?,
        target_bytes: total(&target_operand)?,
        download_bytes: plan.download_bytes,
        install_growth_bytes: plan.install_growth_bytes,
        new_content: plan.new_content.iter().map(|hash| hash.to_hex()).collect(),
        entries: Vec::new(),
    };
    for entry in plan.entries {
        let change = match entry.change {
            PatchChange::Added => {
                result.added += 1;
                "added"
            }
            PatchChange::Removed => {
                result.removed += 1;
                "removed"
            }
            PatchChange::Changed => {
                result.changed += 1;
                "changed"
            }
        };
        let mut fields = Vec::new();
        if let (Some(before), Some(after)) = (&entry.base, &entry.target) {
            for (name, differs) in [
                ("artifact_key", before.key != after.key),
                ("kind", before.kind != after.kind),
                ("schema", before.schema != after.schema),
                ("storage_class", before.storage_class != after.storage_class),
                ("content_hash", before.content_hash != after.content_hash),
                ("bytes", before.byte_len != after.byte_len),
                ("alignment", before.alignment != after.alignment),
                ("dependencies", before.dependencies != after.dependencies),
            ] {
                if differs {
                    fields.push(name);
                }
            }
        }
        result.entries.push(DiffEntry {
            asset_id: entry.asset.0.to_string(),
            change,
            changed_fields: fields,
            base: entry.base.as_ref().map(Into::into),
            target: entry.target.as_ref().map(Into::into),
            content_reused: entry.content_reused,
        });
    }
    result.unchanged = base_operand.entries.len() - result.removed - result.changed;
    Ok(result)
}
impl DiffResult {
    pub fn write_human(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(
            out,
            "base_revision={} target_revision={} base_manifest_hash={} target_manifest_hash={}",
            self.base_revision,
            self.target_revision,
            self.base_manifest_hash,
            self.target_manifest_hash
        )?;
        writeln!(out, "added={} removed={} changed={} unchanged={} base_bytes={} target_bytes={} download_bytes={} install_growth_bytes={}", self.added, self.removed, self.changed, self.unchanged, self.base_bytes, self.target_bytes, self.download_bytes, self.install_growth_bytes)?;
        writeln!(out, "new_content={}", self.new_content.join(","))?;
        for entry in &self.entries {
            writeln!(
                out,
                "asset={} change={} content_reused={} changed_fields={}",
                entry.asset_id,
                entry.change,
                entry.content_reused,
                entry.changed_fields.join(",")
            )?;
            if let Some(base) = &entry.base {
                base.write_human(out, "  base")?;
            }
            if let Some(target) = &entry.target {
                target.write_human(out, "  target")?;
            }
        }
        Ok(())
    }
}
