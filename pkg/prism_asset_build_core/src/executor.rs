//! Deterministic bridge from a build DAG to DDC, CAS, and revision publication.

use crate::cas::CasError;
use crate::ddc::{DdcError, DerivedDataCache};
use crate::graph::{BuildError, BuildGraph};
use crate::hash::{derive_artifact_key, ArtifactKeyInputs, HashError};
use crate::job::JobId;
use crate::transaction::{PublishError, PublishedManifest, RevisionStore};
use crate::types::{ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, ContentHash};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Immutable semantic settings and output contract for one graph node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactSpec {
    /// Logical asset selected by this build node in the published manifest.
    pub asset: AssetId,
    /// Stable processor identity from the project protocol registry.
    pub processor_id: u128,
    /// Processor implementation/schema version included in the cache key.
    pub processor_version: u32,
    /// Canonically encoded processor settings.
    pub canonical_settings: Vec<u8>,
    /// Hash of the selected build profile.
    pub profile_hash: [u8; 32],
    /// Explicit toolchain fingerprint; never inferred from ambient state.
    pub toolchain_fingerprint: Vec<u8>,
    /// Explicit environment fingerprint for declared environmental inputs.
    pub environment_fingerprint: Vec<u8>,
    /// Runtime artifact kind protocol ID.
    pub kind: u128,
    /// Runtime artifact schema version.
    pub schema: u32,
    /// Required output alignment, as a non-zero power of two.
    pub alignment: u32,
}

/// Outcome and cache evidence for a fully published build.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildExecution {
    /// Newly committed manifest and revision.
    pub published: PublishedManifest,
    /// Stable graph job to semantic artifact key mapping.
    pub artifact_keys: BTreeMap<JobId, ArtifactKey>,
    /// Jobs whose bytes were loaded from the verified local DDC/CAS pair.
    pub cache_hits: BTreeSet<JobId>,
}

/// Failure during deterministic execution or publication.
#[derive(Debug)]
pub enum BuildExecutionError {
    /// Graph validation or processor failure.
    Graph(BuildError),
    /// Artifact key derivation rejected an oversized input.
    Hash(HashError),
    /// DDC index access or mutation failed.
    Ddc(DdcError),
    /// CAS access failed for a reason other than a recoverable cache miss/corruption.
    Cas(CasError),
    /// Transactional publication failed.
    Publish(PublishError),
    /// The plan is missing metadata for a graph node.
    MissingSpec(JobId),
    /// The plan contains metadata that has no graph node.
    UnknownSpec(JobId),
    /// More than one graph node publishes the same logical asset.
    DuplicateAsset(AssetId),
    /// Cached descriptor disagrees with this plan's expected artifact contract.
    DescriptorMismatch(JobId),
    /// Output contract could not be represented by a valid descriptor.
    InvalidOutput(JobId, &'static str),
}

impl fmt::Display for BuildExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "build graph failed: {error}"),
            Self::Hash(error) => write!(f, "artifact key derivation failed: {error}"),
            Self::Ddc(error) => write!(f, "DDC failed: {error}"),
            Self::Cas(error) => write!(f, "CAS failed: {error}"),
            Self::Publish(error) => write!(f, "build publication failed: {error}"),
            Self::MissingSpec(id) => write!(f, "build job {} has no artifact spec", id.0),
            Self::UnknownSpec(id) => write!(f, "artifact spec references unknown job {}", id.0),
            Self::DuplicateAsset(asset) => {
                write!(f, "multiple build jobs publish asset {}", asset.0)
            }
            Self::DescriptorMismatch(id) => write!(f, "cached descriptor mismatches job {}", id.0),
            Self::InvalidOutput(id, reason) => {
                write!(f, "job {} has invalid output: {reason}", id.0)
            }
        }
    }
}

impl std::error::Error for BuildExecutionError {}

impl From<BuildError> for BuildExecutionError {
    fn from(value: BuildError) -> Self {
        Self::Graph(value)
    }
}
impl From<HashError> for BuildExecutionError {
    fn from(value: HashError) -> Self {
        Self::Hash(value)
    }
}
impl From<DdcError> for BuildExecutionError {
    fn from(value: DdcError) -> Self {
        Self::Ddc(value)
    }
}
impl From<CasError> for BuildExecutionError {
    fn from(value: CasError) -> Self {
        Self::Cas(value)
    }
}
impl From<PublishError> for BuildExecutionError {
    fn from(value: PublishError) -> Self {
        Self::Publish(value)
    }
}

/// Executes a validated graph using verified DDC hits and publishes all
/// selected outputs as one new manifest revision.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildExecutor {
    /// Version of the ArtifactKey wire/derivation protocol.
    pub protocol_version: u32,
}

impl BuildExecutor {
    /// Execute and publish the graph. DDC entries may be populated before a
    /// later node fails, but `CURRENT` is changed only after every job succeeds.
    pub fn execute(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
    ) -> Result<BuildExecution, BuildExecutionError> {
        let order = graph.execution_order()?;
        validate_specs(&order, specs)?;

        let mut outputs = BTreeMap::<JobId, Vec<u8>>::new();
        let mut descriptors = BTreeMap::<JobId, ArtifactDescriptor>::new();
        let mut artifact_keys = BTreeMap::new();
        let mut cache_hits = BTreeSet::new();

        for id in order {
            let job = graph
                .job(id)
                .expect("execution order only contains graph jobs");
            let spec = &specs[&id];
            let input_keys: Vec<_> = job
                .dependencies()
                .iter()
                .map(|dep| artifact_keys[dep])
                .collect();
            // Runtime interpretation is part of artifact identity too. Keeping
            // it in the canonical setting envelope prevents one semantic key
            // from aliasing two incompatible output contracts.
            let canonical_settings = output_contract_settings(spec);
            let key = derive_artifact_key(&ArtifactKeyInputs {
                protocol_version: self.protocol_version,
                processor_id: spec.processor_id,
                processor_version: spec.processor_version,
                ordered_inputs: &input_keys,
                canonical_settings: &canonical_settings,
                profile_hash: &spec.profile_hash,
                toolchain_fingerprint: &spec.toolchain_fingerprint,
                environment_fingerprint: &spec.environment_fingerprint,
            })?;
            let expected_dependencies = input_keys.clone();
            let cached = ddc.get(key);
            let mut bytes = None;
            if let Some(descriptor) = cached {
                if descriptor.key != key
                    || descriptor.kind != spec.kind
                    || descriptor.schema != spec.schema
                    || descriptor.alignment != spec.alignment
                    || descriptor.dependencies != expected_dependencies
                {
                    return Err(BuildExecutionError::DescriptorMismatch(id));
                }
                match store.cas().read(descriptor.content_hash) {
                    Ok(payload)
                        if payload.len() as u64 == descriptor.byte_len
                            && ContentHash::digest(&payload) == descriptor.content_hash =>
                    {
                        bytes = Some(payload);
                        cache_hits.insert(id);
                    }
                    Ok(_) | Err(CasError::Corrupt { .. }) => {
                        ddc.invalidate(key)?;
                    }
                    Err(CasError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                        ddc.invalidate(key)?;
                    }
                    Err(error) => return Err(BuildExecutionError::Cas(error)),
                }
            }
            let output = match bytes {
                Some(payload) => payload,
                None => {
                    let inputs: Vec<&[u8]> = job
                        .dependencies()
                        .iter()
                        .map(|dep| outputs[dep].as_slice())
                        .collect();
                    job.process(&inputs).map_err(|source| {
                        BuildExecutionError::Graph(BuildError::ProcessorFailed { job: id, source })
                    })?
                }
            };
            if spec.alignment == 0 || !spec.alignment.is_power_of_two() {
                return Err(BuildExecutionError::InvalidOutput(
                    id,
                    "alignment must be a non-zero power of two",
                ));
            }
            let descriptor = ArtifactDescriptor::for_bytes(
                key,
                spec.kind,
                spec.schema,
                spec.alignment,
                expected_dependencies,
                &output,
            );
            if !cache_hits.contains(&id) {
                // CAS durability precedes the DDC index reference.
                let stored_hash = store.cas().put(&output)?;
                if stored_hash != descriptor.content_hash {
                    return Err(BuildExecutionError::InvalidOutput(
                        id,
                        "CAS hash differs from descriptor",
                    ));
                }
                ddc.put(key, descriptor.clone())?;
            }
            artifact_keys.insert(id, key);
            descriptors.insert(id, descriptor);
            outputs.insert(id, output);
        }

        let expected_current = store.current_revision()?;
        let mut transaction = store.begin(revision, expected_current);
        for id in graph.execution_order()? {
            transaction.stage(
                specs[&id].asset,
                descriptors[&id].clone(),
                outputs[&id].clone(),
            )?;
        }
        let published = transaction.commit()?;
        Ok(BuildExecution {
            published,
            artifact_keys,
            cache_hits,
        })
    }
}

fn output_contract_settings(spec: &ArtifactSpec) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + 4 + 4 + 8 + spec.canonical_settings.len());
    bytes.extend_from_slice(b"PRISM-OUTPUT-CONTRACT\0\x01");
    bytes.extend_from_slice(&spec.kind.to_le_bytes());
    bytes.extend_from_slice(&spec.schema.to_le_bytes());
    bytes.extend_from_slice(&spec.alignment.to_le_bytes());
    bytes.extend_from_slice(&(spec.canonical_settings.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&spec.canonical_settings);
    bytes
}

fn validate_specs(
    order: &[JobId],
    specs: &BTreeMap<JobId, ArtifactSpec>,
) -> Result<(), BuildExecutionError> {
    let ids: BTreeSet<_> = order.iter().copied().collect();
    if let Some(id) = specs.keys().find(|id| !ids.contains(id)) {
        return Err(BuildExecutionError::UnknownSpec(*id));
    }
    let mut assets = BTreeSet::new();
    for id in order {
        let spec = specs.get(id).ok_or(BuildExecutionError::MissingSpec(*id))?;
        if !assets.insert(spec.asset) {
            return Err(BuildExecutionError::DuplicateAsset(spec.asset));
        }
        if spec.alignment == 0 || !spec.alignment.is_power_of_two() {
            return Err(BuildExecutionError::InvalidOutput(
                *id,
                "alignment must be a non-zero power of two",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::ProcessorError;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "prism-build-executor-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock must be after unix epoch")
                .as_nanos()
        ))
    }

    fn spec(asset: u128, settings: &[u8]) -> ArtifactSpec {
        ArtifactSpec {
            asset: AssetId(asset),
            processor_id: 17,
            processor_version: 1,
            canonical_settings: settings.to_vec(),
            profile_hash: [3; 32],
            toolchain_fingerprint: b"toolchain-v1".to_vec(),
            environment_fingerprint: b"declared-env".to_vec(),
            kind: 9,
            schema: 1,
            alignment: 16,
        }
    }

    fn one_job(counter: Arc<AtomicUsize>, output: &'static [u8]) -> BuildGraph {
        let mut graph = BuildGraph::new();
        graph
            .add_job(crate::job::BuildJob::new(
                JobId(1),
                [],
                move |_: &[&[u8]]| {
                    counter.fetch_add(1, Ordering::Relaxed);
                    Ok(output.to_vec())
                },
            ))
            .unwrap();
        graph
    }

    fn run(
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        root: &std::path::Path,
        revision: u64,
    ) -> BuildExecution {
        let store = RevisionStore::open(root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        BuildExecutor {
            protocol_version: 1,
        }
        .execute(graph, specs, &store, &mut ddc, BuildRevision(revision))
        .unwrap()
    }

    #[test]
    fn ddc_hit_skips_processor_and_publishes_next_revision() {
        let root = temp_dir("ddc-hit");
        let first_count = Arc::new(AtomicUsize::new(0));
        let first = one_job(Arc::clone(&first_count), b"payload");
        let mut specs = BTreeMap::new();
        specs.insert(JobId(1), spec(42, b"settings"));
        let first_result = run(&first, &specs, &root, 0);
        assert!(first_result.cache_hits.is_empty());
        assert_eq!(first_count.load(Ordering::Relaxed), 1);

        let second_count = Arc::new(AtomicUsize::new(0));
        let second = one_job(Arc::clone(&second_count), b"different processor output");
        let second_result = run(&second, &specs, &root, 1);
        assert_eq!(second_result.cache_hits, [JobId(1)].into_iter().collect());
        assert_eq!(second_count.load(Ordering::Relaxed), 0);
        assert_eq!(second_result.published.manifest.revision, BuildRevision(1));
        assert_eq!(first_result.artifact_keys, second_result.artifact_keys);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn processor_failure_does_not_publish_revision() {
        let root = temp_dir("failure");
        let mut graph = BuildGraph::new();
        graph
            .add_job(crate::job::BuildJob::new(
                JobId(1),
                [],
                |_: &[&[u8]]| Err(ProcessorError::new("broken source")),
            ))
            .unwrap();
        let mut specs = BTreeMap::new();
        specs.insert(JobId(1), spec(42, b"settings"));
        let store = RevisionStore::open(&root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        let result = BuildExecutor {
            protocol_version: 1,
        }
        .execute(&graph, &specs, &store, &mut ddc, BuildRevision(0));
        assert!(matches!(
            result,
            Err(BuildExecutionError::Graph(
                BuildError::ProcessorFailed { .. }
            ))
        ));
        assert_eq!(store.current_revision().unwrap(), None);
        assert!(ddc.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn corrupt_cas_blob_is_rebuilt_instead_of_returned_as_a_hit() {
        let root = temp_dir("corrupt-cas");
        let count = Arc::new(AtomicUsize::new(0));
        let graph = one_job(Arc::clone(&count), b"payload");
        let mut specs = BTreeMap::new();
        specs.insert(JobId(1), spec(42, b"settings"));
        run(&graph, &specs, &root, 0);
        let store = RevisionStore::open(&root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        let execution = BuildExecutor {
            protocol_version: 1,
        }
        .execute(&graph, &specs, &store, &mut ddc, BuildRevision(1))
        .unwrap();
        let key = ddc.get(execution.artifact_keys[&JobId(1)]);
        let descriptor = key.expect("first execution must leave a DDC entry");
        fs::write(store.cas().blob_path(descriptor.content_hash), b"tampered").unwrap();

        let before = count.load(Ordering::Relaxed);
        let rebuilt = run(&graph, &specs, &root, 2);
        assert!(rebuilt.cache_hits.is_empty());
        assert_eq!(count.load(Ordering::Relaxed), before + 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn runtime_output_contract_changes_artifact_identity() {
        let root = temp_dir("contract-key");
        let count = Arc::new(AtomicUsize::new(0));
        let graph = one_job(Arc::clone(&count), b"payload");
        let mut specs = BTreeMap::new();
        specs.insert(JobId(1), spec(42, b"settings"));
        let first = run(&graph, &specs, &root, 0);

        specs.get_mut(&JobId(1)).unwrap().schema = 2;
        let second = run(&graph, &specs, &root, 1);
        assert_ne!(
            first.artifact_keys[&JobId(1)],
            second.artifact_keys[&JobId(1)]
        );
        assert!(second.cache_hits.is_empty());
        assert_eq!(count.load(Ordering::Relaxed), 2);
        fs::remove_dir_all(root).unwrap();
    }
}
