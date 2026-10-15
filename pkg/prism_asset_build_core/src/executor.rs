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
use std::thread;

mod determinism;
mod wave_results;
pub use determinism::DeterminismDifference;

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

impl ArtifactSpec {
    /// Derives the same key used by [`BuildExecutor`] for this output contract.
    pub fn artifact_key(
        &self,
        protocol_version: u32,
        ordered_inputs: &[ArtifactKey],
    ) -> Result<ArtifactKey, HashError> {
        let canonical_settings = output_contract_settings(self);
        derive_artifact_key(&ArtifactKeyInputs {
            protocol_version,
            processor_id: self.processor_id,
            processor_version: self.processor_version,
            ordered_inputs,
            canonical_settings: &canonical_settings,
            profile_hash: &self.profile_hash,
            toolchain_fingerprint: &self.toolchain_fingerprint,
            environment_fingerprint: &self.environment_fingerprint,
        })
    }
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
    /// Two fresh executions of the same job produced different bytes.
    NonDeterministic(DeterminismDifference),
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
            Self::NonDeterministic(difference) => difference.fmt(f),
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
        self.execute_with_limit(graph, specs, store, ddc, revision, 1, false)
    }

    /// Execute independent jobs concurrently, bounded by the host's available
    /// parallelism. DDC mutations and manifest staging remain deterministic.
    pub fn execute_parallel(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
    ) -> Result<BuildExecution, BuildExecutionError> {
        let limit = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        self.execute_parallel_with_limit(graph, specs, store, ddc, revision, limit)
    }

    /// Execute independent jobs concurrently with a caller-selected maximum.
    /// Each ready wave computes DDC/CAS work in parallel; completed results are
    /// applied in ascending `JobId` order, so cache updates, failures, and
    /// publication staging do not depend on thread scheduling.
    pub fn execute_parallel_with_limit(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
    ) -> Result<BuildExecution, BuildExecutionError> {
        self.execute_with_limit(graph, specs, store, ddc, revision, max_concurrency, false)
    }

    /// Bypass every DDC hit and execute each job twice with identical ordered
    /// dependency bytes. Compare exact output bytes before writing that job to
    /// CAS or DDC; publish the first checked output, without a third execution.
    ///
    /// A mismatch, processor error, or worker panic leaves `CURRENT` unchanged.
    /// Successful jobs may still populate CAS/DDC before a later failure. Warm
    /// cache entries are not evidence of determinism and are not compared to
    /// these fresh outputs. This checks local in-process repeatability only;
    /// it does not isolate ambient state or prove cross-machine reproducibility.
    pub fn execute_verified_with_limit(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
    ) -> Result<BuildExecution, BuildExecutionError> {
        self.execute_with_limit(graph, specs, store, ddc, revision, max_concurrency, true)
    }

    fn execute_with_limit(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
        verify_determinism: bool,
    ) -> Result<BuildExecution, BuildExecutionError> {
        if max_concurrency == 0 {
            return Err(BuildExecutionError::Graph(BuildError::InvalidConcurrency));
        }
        let order = graph.execution_order()?;
        validate_specs(&order, specs)?;

        let mut outputs = BTreeMap::<JobId, Vec<u8>>::new();
        let mut descriptors = BTreeMap::<JobId, ArtifactDescriptor>::new();
        let mut artifact_keys = BTreeMap::new();
        let mut cache_hits = BTreeSet::new();

        let mut completed = BTreeSet::new();
        while completed.len() < order.len() {
            let mut ready: Vec<_> = order
                .iter()
                .copied()
                .filter(|id| {
                    !completed.contains(id)
                        && graph.job(*id).is_some_and(|job| {
                            job.dependencies()
                                .iter()
                                .all(|dependency| completed.contains(dependency))
                        })
                })
                .collect();
            ready.sort_unstable();
            let wave: Vec<_> = ready.into_iter().take(max_concurrency).collect();
            if wave.is_empty() {
                // `execution_order` has already validated the graph; this is
                // defensive against a future scheduler regression.
                return Err(BuildExecutionError::Graph(BuildError::Cycle {
                    jobs: order
                        .iter()
                        .copied()
                        .filter(|id| !completed.contains(id))
                        .collect(),
                }));
            }

            let mut work = Vec::with_capacity(wave.len());
            for &id in &wave {
                let job = graph.job(id).expect("ordered job must exist");
                let spec = &specs[&id];
                let input_keys: Vec<_> = job
                    .dependencies()
                    .iter()
                    .map(|dependency| artifact_keys[dependency])
                    .collect();
                let inputs: Vec<&[u8]> = job
                    .dependencies()
                    .iter()
                    .map(|dependency| outputs[dependency].as_slice())
                    .collect();
                let key = spec.artifact_key(self.protocol_version, &input_keys)?;
                work.push((
                    id,
                    key,
                    spec,
                    input_keys,
                    job.processor(),
                    inputs,
                    if verify_determinism {
                        None
                    } else {
                        ddc.get(key)
                    },
                ));
            }

            let results = thread::scope(|scope| {
                let handles = work
                    .into_iter()
                    .map(|(id, key, spec, input_keys, processor, inputs, cached)| {
                        let store = store;
                        (
                            id,
                            scope.spawn(move || {
                                prepare_job(
                                    id,
                                    key,
                                    spec,
                                    input_keys,
                                    processor,
                                    inputs,
                                    cached,
                                    store,
                                    verify_determinism,
                                )
                            }),
                        )
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|(id, handle)| {
                        let result = handle.join().unwrap_or_else(|_| {
                            Err((
                                BuildExecutionError::Graph(BuildError::ProcessorFailed {
                                    job: id,
                                    source: crate::job::ProcessorError::new(
                                        "build worker thread panicked",
                                    ),
                                }),
                                false,
                            ))
                        });
                        (id, result)
                    })
                    .collect::<Vec<_>>()
            });

            // Canonicalize at the side-effect boundary, independently of the
            // collector's arrival/join order. No cache mutation or failure
            // selection may happen before this ordering step.
            for (id, result) in wave_results::ordered(results) {
                let prepared = match result {
                    Ok(prepared) => prepared,
                    Err((error, invalidate_cache)) => {
                        if invalidate_cache {
                            let key = work_key_for(
                                id,
                                graph,
                                &artifact_keys,
                                specs,
                                self.protocol_version,
                            )?;
                            ddc.invalidate(key)?;
                        }
                        return Err(error);
                    }
                };
                if prepared.invalidate_cache {
                    ddc.invalidate(prepared.key)?;
                }
                if !prepared.cache_hit {
                    ddc.put(prepared.key, prepared.descriptor.clone())?;
                } else {
                    cache_hits.insert(id);
                }
                artifact_keys.insert(id, prepared.key);
                descriptors.insert(id, prepared.descriptor);
                outputs.insert(id, prepared.output);
                completed.insert(id);
            }
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

struct PreparedJob {
    key: ArtifactKey,
    descriptor: ArtifactDescriptor,
    output: Vec<u8>,
    cache_hit: bool,
    invalidate_cache: bool,
}

fn work_key_for(
    id: JobId,
    graph: &BuildGraph,
    artifact_keys: &BTreeMap<JobId, ArtifactKey>,
    specs: &BTreeMap<JobId, ArtifactSpec>,
    protocol_version: u32,
) -> Result<ArtifactKey, BuildExecutionError> {
    let input_keys: Vec<_> = graph
        .job(id)
        .expect("ordered job must exist")
        .dependencies()
        .iter()
        .map(|dependency| artifact_keys[dependency])
        .collect();
    Ok(specs[&id].artifact_key(protocol_version, &input_keys)?)
}

fn prepare_job(
    id: JobId,
    key: ArtifactKey,
    spec: &ArtifactSpec,
    input_keys: Vec<ArtifactKey>,
    processor: std::sync::Arc<dyn crate::job::Processor>,
    inputs: Vec<&[u8]>,
    cached: Option<ArtifactDescriptor>,
    store: &RevisionStore,
    verify_determinism: bool,
) -> Result<PreparedJob, (BuildExecutionError, bool)> {
    let mut cached_bytes = None;
    let mut invalidate_cache = false;
    if let Some(descriptor) = cached {
        if descriptor.key != key
            || descriptor.kind != spec.kind
            || descriptor.schema != spec.schema
            || descriptor.alignment != spec.alignment
            || descriptor.dependencies != input_keys
        {
            return Err((BuildExecutionError::DescriptorMismatch(id), false));
        }
        match store.cas().read(descriptor.content_hash) {
            Ok(payload)
                if payload.len() as u64 == descriptor.byte_len
                    && ContentHash::digest(&payload) == descriptor.content_hash =>
            {
                cached_bytes = Some(payload);
            }
            Ok(_) | Err(CasError::Corrupt { .. }) => invalidate_cache = true,
            Err(CasError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                invalidate_cache = true;
            }
            Err(error) => return Err((BuildExecutionError::Cas(error), false)),
        }
    }

    let cache_hit = cached_bytes.is_some();
    let output = match cached_bytes {
        Some(payload) => payload,
        None => {
            let input_refs: Vec<&[u8]> = inputs.iter().copied().collect();
            let output = processor.process(&input_refs).map_err(|source| {
                (
                    BuildExecutionError::Graph(BuildError::ProcessorFailed { job: id, source }),
                    invalidate_cache,
                )
            })?;
            if verify_determinism {
                determinism::verify_second_run(id, key, processor.as_ref(), &input_refs, &output)
                    .map_err(|error| (error, invalidate_cache))?;
            }
            output
        }
    };
    if spec.alignment == 0 || !spec.alignment.is_power_of_two() {
        return Err((
            BuildExecutionError::InvalidOutput(id, "alignment must be a non-zero power of two"),
            invalidate_cache,
        ));
    }
    let descriptor = ArtifactDescriptor::for_bytes(
        key,
        spec.kind,
        spec.schema,
        spec.alignment,
        input_keys,
        &output,
    );
    if !cache_hit {
        // Immutable CAS writes are safe to perform concurrently. The mutable
        // DDC index is updated by the coordinator only after durable CAS put.
        let stored_hash = store
            .cas()
            .put(&output)
            .map_err(|error| (BuildExecutionError::Cas(error), invalidate_cache))?;
        if stored_hash != descriptor.content_hash {
            return Err((
                BuildExecutionError::InvalidOutput(id, "CAS hash differs from descriptor"),
                invalidate_cache,
            ));
        }
    }
    Ok(PreparedJob {
        key,
        descriptor,
        output,
        cache_hit,
        invalidate_cache,
    })
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
    use crate::recipe::ImportProcessor;
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
    fn source_snapshot_and_recipe_drive_a_real_build_executor_chain() {
        let root = temp_dir("source-recipe-chain");
        let source = crate::SourceSnapshot::new("source.bin", b"source bytes".to_vec()).unwrap();
        let source_job = JobId(1);
        let import_job = JobId(2);
        let mut graph = BuildGraph::new();
        graph.add_job(source.build_job(source_job)).unwrap();
        graph
            .add_job(crate::job::BuildJob::new(
                import_job,
                [source_job],
                |inputs: &[&[u8]]| {
                    let [source] = inputs else {
                        return Err(ProcessorError::new("importer expects one source input"));
                    };
                    let mut output = source.to_vec();
                    output.extend_from_slice(b"|imported");
                    Ok(output)
                },
            ))
            .unwrap();

        let recipe = crate::ImportRecipe {
            importer_id: 0x494d504f52544552,
            importer_version: 1,
            canonical_settings: b"{\"mode\":\"test\"}".to_vec(),
            profile_hash: [3; 32],
            toolchain_fingerprint: b"toolchain-v1".to_vec(),
            environment_fingerprint: b"test-platform".to_vec(),
        };
        let mut specs = BTreeMap::new();
        specs.insert(source_job, source.artifact_spec(AssetId(1)));
        let import_spec = recipe.artifact_spec(AssetId(2), 0x494d504f5254, 1, 16);
        specs.insert(import_job, import_spec.clone());

        let store = RevisionStore::open(&root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        let execution = BuildExecutor {
            protocol_version: 1,
        }
        .execute(&graph, &specs, &store, &mut ddc, BuildRevision(0))
        .unwrap();

        assert_eq!(execution.published.manifest.revision, BuildRevision(0));
        assert_eq!(execution.published.manifest.entries.len(), 2);
        assert_eq!(execution.artifact_keys[&source_job], source.artifact_key());
        assert_eq!(
            execution.artifact_keys[&import_job],
            import_spec
                .artifact_key(1, &[execution.artifact_keys[&source_job]])
                .unwrap()
        );
        assert_eq!(
            store
                .cas()
                .read(execution.published.manifest.entries[&AssetId(2)].content_hash)
                .unwrap(),
            b"source bytes|imported"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn png_importer_publishes_a_canonical_image_artifact() {
        let root = temp_dir("png-import-chain");
        let mut png_bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png_bytes, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[9, 8, 7]).unwrap();
        }
        let source = crate::SourceSnapshot::new("textures/one.png", png_bytes).unwrap();
        let source_job = JobId(10);
        let import_job = JobId(11);
        let mut graph = BuildGraph::new();
        graph.add_job(source.build_job(source_job)).unwrap();
        let recipe = crate::ImportRecipe {
            importer_id: crate::import::CanonicalPngProcessor::IMPORTER_ID,
            importer_version: 1,
            canonical_settings: b"rgba8".to_vec(),
            profile_hash: [1; 32],
            toolchain_fingerprint: b"png-0.18".to_vec(),
            environment_fingerprint: b"portable".to_vec(),
        };
        let plan = recipe
            .plan(
                AssetId(11),
                import_job,
                source_job,
                crate::import::CanonicalPngProcessor,
                crate::import::IMAGE_ARTIFACT_KIND,
                crate::import::IMAGE_ARTIFACT_SCHEMA,
                4,
            )
            .unwrap();
        graph.add_job(plan.job).unwrap();
        let mut specs = BTreeMap::new();
        specs.insert(source_job, source.artifact_spec(AssetId(10)));
        specs.insert(import_job, plan.spec);
        let store = RevisionStore::open(&root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        let execution = BuildExecutor {
            protocol_version: 1,
        }
        .execute(&graph, &specs, &store, &mut ddc, BuildRevision(0))
        .unwrap();
        let bytes = store
            .cas()
            .read(execution.published.manifest.entries[&AssetId(11)].content_hash)
            .unwrap();
        let image = crate::import::ImageArtifact::decode(&bytes).unwrap();
        assert_eq!((image.width, image.height), (1, 1));
        assert_eq!(image.rgba8, vec![9, 8, 7, 255]);
        assert_eq!(
            execution.artifact_keys[&import_job],
            recipe
                .artifact_key_for_output(
                    1,
                    &source,
                    &[],
                    crate::import::IMAGE_ARTIFACT_KIND,
                    crate::import::IMAGE_ARTIFACT_SCHEMA,
                    4,
                )
                .unwrap()
        );

        let second = BuildExecutor {
            protocol_version: 1,
        }
        .execute(&graph, &specs, &store, &mut ddc, BuildRevision(1))
        .unwrap();
        assert_eq!(
            second.cache_hits,
            [source_job, import_job].into_iter().collect()
        );
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
        // `RevisionStore::open` intentionally fails closed when CURRENT names
        // a corrupt payload. Keep the already-open store here so this test
        // exercises the executor's recoverable DDC/CAS cache-rebuild path
        // without weakening publication-boundary validation.
        let rebuilt = BuildExecutor {
            protocol_version: 1,
        }
        .execute(&graph, &specs, &store, &mut ddc, BuildRevision(2))
        .unwrap();
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

    #[test]
    fn parallel_executor_runs_ready_jobs_and_preserves_outputs() {
        let root = temp_dir("parallel");
        let serial_root = temp_dir("parallel-serial-reference");
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut graph = BuildGraph::new();
        for id in [1_u128, 2, 3, 4] {
            let active = Arc::clone(&active);
            let peak = Arc::clone(&peak);
            graph
                .add_job(crate::job::BuildJob::new(
                    JobId(id),
                    [],
                    move |_: &[&[u8]]| {
                        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(current, Ordering::SeqCst);
                        thread::sleep(std::time::Duration::from_millis(20));
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(vec![id as u8])
                    },
                ))
                .unwrap();
        }
        let mut specs = BTreeMap::new();
        for id in [1_u128, 2, 3, 4] {
            specs.insert(JobId(id), spec(100 + id, &[id as u8]));
        }
        let store = RevisionStore::open(&root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        let execution = BuildExecutor {
            protocol_version: 1,
        }
        .execute_parallel_with_limit(&graph, &specs, &store, &mut ddc, BuildRevision(0), 4)
        .unwrap();
        let serial = run(&graph, &specs, &serial_root, 0);
        assert!(peak.load(Ordering::SeqCst) > 1);
        assert_eq!(execution.published.manifest.entries.len(), 4);
        assert_eq!(execution.artifact_keys.len(), 4);
        assert_eq!(execution.artifact_keys, serial.artifact_keys);
        assert_eq!(
            execution.published.manifest.canonical_bytes(),
            serial.published.manifest.canonical_bytes()
        );
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(serial_root).unwrap();
    }

    #[test]
    fn parallel_executor_reports_smallest_failed_job_and_does_not_publish() {
        let root = temp_dir("parallel-failure");
        let mut graph = BuildGraph::new();
        for id in [2_u128, 5] {
            graph
                .add_job(crate::job::BuildJob::new(
                    JobId(id),
                    [],
                    move |_: &[&[u8]]| {
                        if id == 2 {
                            thread::sleep(std::time::Duration::from_millis(30));
                        }
                        Err(ProcessorError::new(format!("failure-{id}")))
                    },
                ))
                .unwrap();
        }
        let mut specs = BTreeMap::new();
        specs.insert(JobId(2), spec(102, b"two"));
        specs.insert(JobId(5), spec(105, b"five"));
        let store = RevisionStore::open(&root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        let result = BuildExecutor {
            protocol_version: 1,
        }
        .execute_parallel_with_limit(&graph, &specs, &store, &mut ddc, BuildRevision(0), 2);
        assert!(matches!(
            result,
            Err(BuildExecutionError::Graph(BuildError::ProcessorFailed {
                job: JobId(2),
                ..
            }))
        ));
        assert_eq!(store.current_revision().unwrap(), None);
        assert!(ddc.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parallel_executor_reuses_verified_ddc_and_cas_entries() {
        let root = temp_dir("parallel-cache-hit");
        let first_count = Arc::new(AtomicUsize::new(0));
        let mut graph = BuildGraph::new();
        for id in [1_u128, 2, 3] {
            let count = Arc::clone(&first_count);
            graph
                .add_job(crate::job::BuildJob::new(
                    JobId(id),
                    [],
                    move |_: &[&[u8]]| {
                        count.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![id as u8, 0xA5])
                    },
                ))
                .unwrap();
        }
        let mut specs = BTreeMap::new();
        for id in [1_u128, 2, 3] {
            specs.insert(JobId(id), spec(200 + id, &[id as u8]));
        }
        let store = RevisionStore::open(&root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        let executor = BuildExecutor {
            protocol_version: 1,
        };
        executor
            .execute_parallel_with_limit(&graph, &specs, &store, &mut ddc, BuildRevision(0), 3)
            .unwrap();
        assert_eq!(first_count.load(Ordering::SeqCst), 3);

        let second_count = Arc::new(AtomicUsize::new(0));
        let mut cached_graph = BuildGraph::new();
        for id in [1_u128, 2, 3] {
            let count = Arc::clone(&second_count);
            cached_graph
                .add_job(crate::job::BuildJob::new(
                    JobId(id),
                    [],
                    move |_: &[&[u8]]| {
                        count.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![0xFF])
                    },
                ))
                .unwrap();
        }
        let second = executor
            .execute_parallel_with_limit(
                &cached_graph,
                &specs,
                &store,
                &mut ddc,
                BuildRevision(1),
                3,
            )
            .unwrap();
        assert_eq!(second_count.load(Ordering::SeqCst), 0);
        assert_eq!(
            second.cache_hits,
            [JobId(1), JobId(2), JobId(3)].into_iter().collect()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parallel_executor_rejects_zero_concurrency_without_mutating_cache() {
        let root = temp_dir("parallel-zero-concurrency");
        let graph = one_job(Arc::new(AtomicUsize::new(0)), b"payload");
        let mut specs = BTreeMap::new();
        specs.insert(JobId(1), spec(42, b"settings"));
        let store = RevisionStore::open(&root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        let result = (BuildExecutor {
            protocol_version: 1,
        })
        .execute_parallel_with_limit(
            &graph,
            &specs,
            &store,
            &mut ddc,
            BuildRevision(0),
            0,
        );
        assert!(matches!(
            result,
            Err(BuildExecutionError::Graph(BuildError::InvalidConcurrency))
        ));
        assert!(ddc.is_empty());
        assert_eq!(store.current_revision().unwrap(), None);
        fs::remove_dir_all(root).unwrap();
    }
}
