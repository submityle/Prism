//! Deterministic bridge from a build DAG to DDC, CAS, and revision publication.

use crate::cas::CasError;
use crate::ddc::{DdcError, DerivedDataCache};
use crate::dirty::{
    read_bounded, sync_directory, BuildInputSnapshot, Cursor, DirtySet, InvalidationPlan,
    RebuildExplanationError,
};
use crate::graph::{BuildError, BuildGraph};
use crate::hash::{derive_artifact_key, ArtifactKeyInputs, HashError};
use crate::job::{
    admit_materialized_inputs, materialize_inputs, JobId, JobProcessor, ProcessorInput,
};
use crate::journal::read_committed_binding as input_binding;
use crate::resources::{BuildResources, ResourceError};
use crate::transaction::{CommitError, PublishError, PublishedManifest, RevisionStore};
use crate::types::{
    ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, ContentHash, Manifest, ManifestEntry,
    TransactionId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::thread;

mod determinism;
mod output_limits;
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
    /// Coordinator-owned payload memory. Persistent execution retains only
    /// descriptors, so these counters are zero; worker/adapter heaps are excluded.
    pub output_memory: BuildOutputMemoryStats,
}

/// Coordinator payload high-water marks, not a process-wide memory budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BuildOutputMemoryStats {
    /// Largest sum of retained output lengths after a completed wave.
    pub peak_retained_bytes: u64,
    /// Largest number of retained outputs after a completed wave.
    pub peak_retained_outputs: usize,
}

/// Failure during deterministic execution or publication.
#[derive(Debug)]
pub enum BuildExecutionError {
    /// Admission failed before any processor ran.
    Resources(ResourceError),
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
    /// The incremental plan contains a job that is not in this graph.
    UnknownDirtyJob(JobId),
    /// More than one graph node publishes the same logical asset.
    DuplicateAsset(AssetId),
    /// Cached descriptor disagrees with this plan's expected artifact contract.
    DescriptorMismatch(JobId),
    /// Output contract could not be represented by a valid descriptor.
    InvalidOutput(JobId, &'static str),
    /// Two fresh executions of the same job produced different bytes.
    NonDeterministic(DeterminismDifference),
    /// Semantic input/explanation persistence or validation failed.
    Explanation(RebuildExplanationError),
}

impl fmt::Display for BuildExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resources(error) => write!(f, "build resource admission failed: {error}"),
            Self::Graph(error) => write!(f, "build graph failed: {error}"),
            Self::Hash(error) => write!(f, "artifact key derivation failed: {error}"),
            Self::Ddc(error) => write!(f, "DDC failed: {error}"),
            Self::Cas(error) => write!(f, "CAS failed: {error}"),
            Self::Publish(error) => write!(f, "build publication failed: {error}"),
            Self::MissingSpec(id) => write!(f, "build job {} has no artifact spec", id.0),
            Self::UnknownSpec(id) => write!(f, "artifact spec references unknown job {}", id.0),
            Self::UnknownDirtyJob(id) => {
                write!(f, "incremental plan references unknown job {}", id.0)
            }
            Self::DuplicateAsset(asset) => {
                write!(f, "multiple build jobs publish asset {}", asset.0)
            }
            Self::DescriptorMismatch(id) => write!(f, "cached descriptor mismatches job {}", id.0),
            Self::InvalidOutput(id, reason) => {
                write!(f, "job {} has invalid output: {reason}", id.0)
            }
            Self::NonDeterministic(difference) => difference.fmt(f),
            Self::Explanation(error) => error.fmt(f),
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

impl From<RebuildExplanationError> for BuildExecutionError {
    fn from(value: RebuildExplanationError) -> Self {
        Self::Explanation(value)
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
    /// Builds against a captured committed baseline and persists semantic inputs
    /// and their actual invalidation plan before publication. The transaction
    /// journal binds the metadata digest and manifest together. Dirty jobs bypass
    /// DDC; unchanged jobs retain the usual verified cache reuse.
    pub fn execute_explained_with_limit(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        inputs: &BuildInputSnapshot,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
        verify_determinism: bool,
    ) -> Result<BuildExecution, BuildExecutionError> {
        self.execute_explained_with_resources(
            graph,
            specs,
            inputs,
            store,
            ddc,
            revision,
            max_concurrency,
            BuildResources::UNLIMITED,
            verify_determinism,
        )
    }

    /// Persists invalidation explanations while enforcing declared concurrent
    /// resource demand. Admission validates the entire graph before executing
    /// processors; the budget is not a sandbox for processor allocations.
    pub fn execute_explained_with_resources(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        inputs: &BuildInputSnapshot,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
        budget: BuildResources,
        verify_determinism: bool,
    ) -> Result<BuildExecution, BuildExecutionError> {
        self.execute_with_limit(
            graph,
            specs,
            store,
            ddc,
            revision,
            max_concurrency,
            verify_determinism,
            budget,
            None,
            Some(inputs),
        )
    }

    /// Reads only existing files. A sidecar alone, Prepare, CURRENT without a
    /// Commit, missing metadata, or a digest/manifest mismatch is never accepted.
    /// No store initialization, recovery, journal truncation or DDC open occurs.
    pub fn load_committed_build_inputs(
        &self,
        root: impl AsRef<Path>,
        revision: BuildRevision,
    ) -> Result<CommittedBuildInputs, RebuildExplanationError> {
        let root = root.as_ref();
        let binding = input_binding(root, revision)?.ok_or(RebuildExplanationError::Invalid(
            "revision has no durable committed input binding",
        ))?;
        if binding.publication_hash.is_some() {
            return Err(RebuildExplanationError::Invalid(
                "input record does not bind a runtime publication",
            ));
        }
        let bytes = read_bounded(
            &input_record_path(root, revision, binding.id),
            MAX_INPUT_RECORD_BYTES,
        )?;
        if input_transaction_id(&bytes) != binding.id {
            return Err(RebuildExplanationError::Invalid(
                "input metadata digest differs from committed transaction",
            ));
        }
        let record = CommittedBuildInputs::decode(&bytes)?;
        if record.revision != revision
            || record.manifest_hash != binding.manifest_hash
            || record.previous_revision != binding.previous_revision
        {
            return Err(RebuildExplanationError::Invalid(
                "input metadata does not bind committed revision/manifest",
            ));
        }
        let manifest = read_bounded(
            &root
                .join("revisions")
                .join(format!("{}.manifest", revision.0)),
            MAX_INPUT_RECORD_BYTES,
        )?;
        let header = b"PRISM-MANIFEST\0\x02";
        if ContentHash::digest(&manifest) != binding.manifest_hash
            || !manifest.starts_with(header)
            || manifest.get(header.len()..header.len() + 8)
                != Some(revision.0.to_le_bytes().as_slice())
        {
            return Err(RebuildExplanationError::Invalid(
                "committed manifest differs from input metadata/journal",
            ));
        }
        Ok(record)
    }

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
        self.execute_with_limit(
            graph,
            specs,
            store,
            ddc,
            revision,
            1,
            false,
            BuildResources::UNLIMITED,
            None,
            None,
        )
    }

    /// Execute a deterministic incremental build serially.
    ///
    /// Jobs in `dirty` are always processed again, even when their semantic
    /// key is already present in DDC. All other jobs use the same verified
    /// DDC/CAS reuse path as [`Self::execute`]. The complete manifest is still
    /// staged and published as one atomic revision after every job succeeds.
    pub fn execute_incremental(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        dirty: &DirtySet,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
    ) -> Result<BuildExecution, BuildExecutionError> {
        self.execute_incremental_with_resources(
            graph,
            specs,
            dirty,
            store,
            ddc,
            revision,
            1,
            BuildResources::UNLIMITED,
            false,
        )
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
        self.execute_with_limit(
            graph,
            specs,
            store,
            ddc,
            revision,
            max_concurrency,
            false,
            BuildResources::UNLIMITED,
            None,
            None,
        )
    }

    /// Execute a deterministic incremental build with a caller-selected
    /// worker limit. Ready waves, dependency order, cache mutations, failure
    /// selection, and publication staging are identical to the full parallel
    /// executor; only the DDC policy differs for jobs in `dirty`.
    pub fn execute_incremental_parallel_with_limit(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        dirty: &DirtySet,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
    ) -> Result<BuildExecution, BuildExecutionError> {
        self.execute_incremental_with_resources(
            graph,
            specs,
            dirty,
            store,
            ddc,
            revision,
            max_concurrency,
            BuildResources::UNLIMITED,
            false,
        )
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
        self.execute_with_limit(
            graph,
            specs,
            store,
            ddc,
            revision,
            max_concurrency,
            true,
            BuildResources::UNLIMITED,
            None,
            None,
        )
    }

    /// Run an incremental build through the existing two-run determinism
    /// gate. As with [`Self::execute_verified_with_limit`], the gate bypasses
    /// every cache hit so every node is executed twice; `dirty` remains an
    /// explicit plan and is validated against the graph.
    pub fn execute_incremental_verified_with_limit(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        dirty: &DirtySet,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
    ) -> Result<BuildExecution, BuildExecutionError> {
        self.execute_incremental_with_resources(
            graph,
            specs,
            dirty,
            store,
            ddc,
            revision,
            max_concurrency,
            BuildResources::UNLIMITED,
            true,
        )
    }

    /// Bound simultaneous declared demand as well as worker count. All job
    /// requests are validated before execution, including warm-cache jobs.
    /// Ready jobs are admitted by ascending ID, skipping jobs which do not
    /// fit the remaining wave. Every finite graph makes forward progress.
    /// Working-set estimates exclude retained dependency/output bytes; this
    /// is admission control, not an allocator-enforced process memory limit.
    pub fn execute_with_resources(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
        budget: BuildResources,
        verify_determinism: bool,
    ) -> Result<BuildExecution, BuildExecutionError> {
        self.execute_with_limit(
            graph,
            specs,
            store,
            ddc,
            revision,
            max_concurrency,
            verify_determinism,
            budget,
            None,
            None,
        )
    }

    /// Execute an incremental build with worker and declared resource
    /// admission limits. Resource validation is unchanged from the full
    /// executor, including validation of cache-reused jobs.
    pub fn execute_incremental_with_resources(
        &self,
        graph: &BuildGraph,
        specs: &BTreeMap<JobId, ArtifactSpec>,
        dirty: &DirtySet,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
        budget: BuildResources,
        verify_determinism: bool,
    ) -> Result<BuildExecution, BuildExecutionError> {
        self.execute_with_limit(
            graph,
            specs,
            store,
            ddc,
            revision,
            max_concurrency,
            verify_determinism,
            budget,
            Some(dirty),
            None,
        )
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
        budget: BuildResources,
        dirty: Option<&DirtySet>,
        inputs: Option<&BuildInputSnapshot>,
    ) -> Result<BuildExecution, BuildExecutionError> {
        if max_concurrency == 0 {
            return Err(BuildExecutionError::Graph(BuildError::InvalidConcurrency));
        }
        let order = graph.execution_order()?;
        validate_specs(&order, specs)?;
        if let Some(dirty) = dirty {
            validate_dirty(&order, dirty)?;
        }
        // Freeze the optimistic baseline before executing processors, never
        // recapture CURRENT after work has raced with another publisher.
        let expected_current = store.build_baseline()?;
        let next = match expected_current {
            None => Some(0),
            Some(current) => current.0.checked_add(1),
        };
        if next != Some(revision.0) {
            return Err(
                PublishError::Invalid("revision must be the next monotonic revision").into(),
            );
        }
        let root = store
            .cas()
            .root()
            .parent()
            .ok_or(PublishError::Invalid("CAS has no store root"))?;
        let explanation = if let Some(inputs) = inputs {
            // Reject oversized or malformed evidence before processors or cache
            // writes. A successful build must have a decodable durable record.
            BuildInputSnapshot::from_canonical_bytes(&inputs.canonical_bytes())?;
            let previous = match expected_current {
                Some(revision) => self
                    .load_committed_build_inputs(root, revision)?
                    .snapshot
                    .retained_for_graph(graph),
                None => {
                    // Even an empty store can retain aborted journal history.
                    // Reject history the read-only query cannot decode before
                    // publishing a revision that why could never retrieve.
                    if input_binding(root, revision)
                        .map_err(RebuildExplanationError::from)?
                        .is_some()
                    {
                        return Err(RebuildExplanationError::Invalid(
                            "journal already commits the requested cold revision",
                        )
                        .into());
                    }
                    BuildInputSnapshot::new()
                }
            };
            let plan = previous.invalidation_plan(inputs, graph).map_err(|_| {
                RebuildExplanationError::Invalid(
                    "semantic inputs do not match validated build graph",
                )
            })?;
            Some(plan)
        } else {
            None
        };
        let dirty = explanation
            .as_ref()
            .map(InvalidationPlan::dirty_set)
            .or(dirty);
        if let Some(dirty) = dirty {
            validate_dirty(&order, dirty)?;
        }
        if budget.cpu_threads == 0 {
            return Err(BuildExecutionError::Resources(ResourceError::ZeroCpuBudget));
        }
        let mut remaining_dependencies = BTreeMap::new();
        let mut dependents: BTreeMap<JobId, Vec<JobId>> = BTreeMap::new();
        let mut ready = BTreeSet::new();
        for &id in &order {
            let job = graph.job(id).expect("validated job exists");
            let request = job.resources();
            job.processor()
                .validate_admission(&specs[&id], request)
                .map_err(|source| {
                    BuildExecutionError::Graph(BuildError::ProcessorFailed { job: id, source })
                })?;
            if request.cpu_threads == 0 {
                return Err(BuildExecutionError::Resources(
                    ResourceError::ZeroCpuRequest(id),
                ));
            }
            if !request.fits(budget) {
                return Err(BuildExecutionError::Resources(
                    ResourceError::ExceedsBudget {
                        job: id,
                        requested: request,
                        budget,
                    },
                ));
            }
            remaining_dependencies.insert(id, job.dependencies().len());
            if job.dependencies().is_empty() {
                ready.insert(id);
            }
            for &dependency in job.dependencies() {
                dependents.entry(dependency).or_default().push(id);
            }
        }

        // Jobs exchange durable CAS descriptors, never retained payload Vecs.
        // Protect every unpublished output through publication or failure.
        let _output_lease = store.protect_build_outputs()?;
        let output_memory = BuildOutputMemoryStats::default();
        let mut descriptors = BTreeMap::<JobId, ArtifactDescriptor>::new();
        let mut artifact_keys = BTreeMap::new();
        let mut cache_hits = BTreeSet::new();

        let mut completed = BTreeSet::new();
        while completed.len() < order.len() {
            let mut available = budget;
            let wave: Vec<_> = ready
                .iter()
                .copied()
                .filter(|id| {
                    graph
                        .job(*id)
                        .expect("ready job exists")
                        .resources()
                        .reserve(&mut available)
                })
                .take(max_concurrency)
                .collect();
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
                let inputs: Vec<ArtifactDescriptor> = job
                    .dependencies()
                    .iter()
                    .map(|dependency| descriptors[dependency].clone())
                    .collect();
                let key = spec.artifact_key(self.protocol_version, &input_keys)?;
                let processor = job.processor();
                processor.validate(spec, &inputs).map_err(|source| {
                    BuildExecutionError::Graph(BuildError::ProcessorFailed { job: id, source })
                })?;
                let force_rebuild = dirty.is_some_and(|dirty| dirty.contains(id));
                work.push((
                    id,
                    key,
                    spec,
                    input_keys,
                    processor.clone(),
                    job.max_output_bytes(),
                    job.max_materialized_input_bytes(),
                    inputs,
                    if verify_determinism || force_rebuild || !processor.cacheable() {
                        None
                    } else {
                        ddc.get(key)
                    },
                ));
            }

            let results = thread::scope(|scope| {
                let handles = work
                    .into_iter()
                    .map(
                        |(
                            id,
                            key,
                            spec,
                            input_keys,
                            processor,
                            max_output_bytes,
                            max_materialized_input_bytes,
                            inputs,
                            cached,
                        )| {
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
                                        max_output_bytes,
                                        max_materialized_input_bytes,
                                        inputs,
                                        cached,
                                        store,
                                        verify_determinism,
                                    )
                                }),
                            )
                        },
                    )
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
                if !prepared.cache_hit
                    && graph
                        .job(id)
                        .expect("completed job exists")
                        .processor()
                        .cacheable()
                {
                    ddc.put(prepared.key, prepared.descriptor.clone())?;
                } else if prepared.cache_hit {
                    cache_hits.insert(id);
                }
                artifact_keys.insert(id, prepared.key);
                descriptors.insert(id, prepared.descriptor);
                completed.insert(id);
                ready.remove(&id);
                if let Some(children) = dependents.get(&id) {
                    for child in children {
                        let pending = remaining_dependencies
                            .get_mut(child)
                            .expect("known dependent");
                        *pending -= 1;
                        if *pending == 0 {
                            ready.insert(*child);
                        }
                    }
                }
            }
        }

        for &id in &order {
            graph
                .job(id)
                .expect("validated job exists")
                .processor()
                .check()
                .map_err(|source| {
                    BuildExecutionError::Graph(BuildError::ProcessorFailed { job: id, source })
                })?;
        }
        let mut transaction = if let (Some(snapshot), Some(plan)) = (inputs, explanation) {
            let manifest = Manifest::new(
                revision,
                order.iter().map(|id| ManifestEntry {
                    asset: specs[id].asset,
                    descriptor: descriptors[id].clone(),
                }),
            )
            .map_err(PublishError::Invalid)?;
            let manifest_bytes = manifest.canonical_bytes();
            if manifest_bytes.len() > MAX_INPUT_RECORD_BYTES {
                return Err(RebuildExplanationError::Invalid(
                    "explained manifest exceeds maximum encoded size",
                )
                .into());
            }
            let record = CommittedBuildInputs {
                revision,
                previous_revision: expected_current,
                manifest_hash: ContentHash::digest(&manifest_bytes),
                snapshot: snapshot.clone(),
                plan,
            };
            let bytes = record.encode();
            // Validate our own bounded representation before any commit can
            // make it visible. Publication never depends on a post-commit write.
            CommittedBuildInputs::decode(&bytes)?;
            let id = input_transaction_id(&bytes);
            persist_input_record(&input_record_path(root, revision, id), &bytes)?;
            store.begin_with_id(revision, expected_current, id)
        } else {
            store.begin(revision, expected_current)
        };
        for &id in &order {
            transaction.stage_existing(specs[&id].asset, descriptors[&id].clone())?;
        }
        let published = transaction
            .commit_checked(|| {
                for &id in &order {
                    graph
                        .job(id)
                        .expect("validated job exists")
                        .processor()
                        .check()
                        .map_err(|source| BuildError::ProcessorFailed { job: id, source })?;
                }
                Ok(())
            })
            .map_err(|error| match error {
                CommitError::Publish(error) => BuildExecutionError::Publish(error),
                CommitError::Rejected(error) => BuildExecutionError::Graph(error),
            })?;
        Ok(BuildExecution {
            published,
            artifact_keys,
            cache_hits,
            output_memory,
        })
    }
}

/// Semantic evidence bound by the committed transaction and exact manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct CommittedBuildInputs {
    /// Committed revision.
    pub revision: BuildRevision,
    /// Baseline captured before any processors executed.
    pub previous_revision: Option<BuildRevision>,
    /// Digest of the exact canonical committed manifest.
    pub manifest_hash: ContentHash,
    /// Semantic inputs actually compared by this execution.
    pub snapshot: BuildInputSnapshot,
    /// Computed direct changes and successor closure.
    pub plan: InvalidationPlan,
}

const INPUT_RECORD_HEADER: &[u8] = b"PRISM-COMMITTED-BUILD-INPUTS\0\x01";
const MAX_INPUT_RECORD_BYTES: usize = 16 * 1024 * 1024;

impl CommittedBuildInputs {
    fn encode(&self) -> Vec<u8> {
        let mut bytes = INPUT_RECORD_HEADER.to_vec();
        bytes.extend_from_slice(&self.revision.0.to_le_bytes());
        bytes.push(u8::from(self.previous_revision.is_some()));
        bytes.extend_from_slice(
            &self
                .previous_revision
                .map_or(0, |revision| revision.0)
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&self.manifest_hash.0);
        for value in [self.snapshot.canonical_bytes(), self.plan.canonical_bytes()] {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&value);
        }
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<Self, RebuildExplanationError> {
        let mut cursor = Cursor::bounded(bytes)?;
        if cursor.take(INPUT_RECORD_HEADER.len())? != INPUT_RECORD_HEADER {
            return Err(RebuildExplanationError::Invalid(
                "unknown committed input header",
            ));
        }
        let revision = BuildRevision(cursor.u64()?);
        let marker = cursor.u8()?;
        let previous = cursor.u64()?;
        let previous_revision = match (marker, previous) {
            (0, 0) if revision.0 == 0 => None,
            (1, previous) if previous.checked_add(1) == Some(revision.0) => {
                Some(BuildRevision(previous))
            }
            _ => {
                return Err(RebuildExplanationError::Invalid(
                    "invalid input baseline revision",
                ))
            }
        };
        let manifest_hash = ContentHash(cursor.array32()?);
        let snapshot = BuildInputSnapshot::from_canonical_bytes(cursor.blob()?)?;
        let plan = InvalidationPlan::from_canonical_bytes(cursor.blob()?)?;
        let record = Self {
            revision,
            previous_revision,
            manifest_hash,
            snapshot,
            plan,
        };
        if !cursor.is_empty() || record.encode() != bytes {
            return Err(RebuildExplanationError::Invalid(
                "noncanonical committed inputs",
            ));
        }
        Ok(record)
    }
}

fn input_transaction_id(bytes: &[u8]) -> TransactionId {
    // The existing durable journal stores 128-bit transaction identities.
    // Domain separation is provided by the versioned input record header.
    TransactionId(ContentHash::digest(bytes).0[..16].try_into().unwrap())
}

fn input_record_path(root: &Path, revision: BuildRevision, id: TransactionId) -> PathBuf {
    root.join("revisions")
        .join(format!("{}.inputs-{}.bin", revision.0, id.to_hex()))
}

fn persist_input_record(path: &Path, bytes: &[u8]) -> Result<(), RebuildExplanationError> {
    let parent = path
        .parent()
        .ok_or(RebuildExplanationError::Invalid("input path has no parent"))?;
    // RevisionStore::open already durably created this directory. Immutable
    // publication prevents a losing process from replacing the winner's data.
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    if let Err(error) = temporary.persist_noclobber(path) {
        if error.error.kind() != io::ErrorKind::AlreadyExists {
            return Err(error.error.into());
        }
        if read_bounded(path, MAX_INPUT_RECORD_BYTES)? != bytes {
            return Err(RebuildExplanationError::Invalid(
                "immutable input metadata conflict",
            ));
        }
        fs::File::open(path)?.sync_all()?;
    }
    sync_directory(parent)?;
    Ok(())
}

struct PreparedJob {
    key: ArtifactKey,
    descriptor: ArtifactDescriptor,
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
    processor: JobProcessor,
    max_output_bytes: u64,
    max_materialized_input_bytes: u64,
    inputs: Vec<ArtifactDescriptor>,
    cached: Option<ArtifactDescriptor>,
    store: &RevisionStore,
    verify_determinism: bool,
) -> Result<PreparedJob, (BuildExecutionError, bool)> {
    // Input slots remain ordered and may repeat for key derivation and stream
    // execution. A published descriptor records each direct dependency once,
    // in first-occurrence order, as required by the publication graph.
    let mut seen = BTreeSet::new();
    let dependency_keys: Vec<_> = input_keys
        .into_iter()
        .filter(|key| seen.insert(*key))
        .collect();
    processor.validate(spec, &inputs).map_err(|source| {
        (
            BuildExecutionError::Graph(BuildError::ProcessorFailed { job: id, source }),
            false,
        )
    })?;
    let mut invalidate_cache = false;
    if let Some(descriptor) = cached {
        if descriptor.key != key
            || descriptor.kind != spec.kind
            || descriptor.schema != spec.schema
            || descriptor.alignment != spec.alignment
            || descriptor.dependencies != dependency_keys
        {
            return Err((BuildExecutionError::DescriptorMismatch(id), false));
        }
        let valid = output_limits::verify_cached(id, store.cas(), &descriptor, max_output_bytes)
            .map_err(|error| (error, false))?;
        if valid {
            return Ok(PreparedJob {
                key,
                descriptor,
                cache_hit: true,
                invalidate_cache: false,
            });
        }
        invalidate_cache = true;
    }
    let mut output = run_processor(
        id,
        &processor,
        &inputs,
        store,
        max_output_bytes,
        max_materialized_input_bytes,
    )
    .map_err(|error| (error, invalidate_cache))?;
    if verify_determinism {
        let mut second = run_processor(
            id,
            &processor,
            &inputs,
            store,
            max_output_bytes,
            max_materialized_input_bytes,
        )
        .map_err(|error| (error, invalidate_cache))?;
        determinism::verify_streams(id, key, &mut output, &mut second)
            .map_err(|error| (error, invalidate_cache))?;
    }
    // Both executions have passed before any candidate output enters CAS.
    processor.check().map_err(|source| {
        (
            BuildExecutionError::Graph(BuildError::ProcessorFailed { job: id, source }),
            invalidate_cache,
        )
    })?;
    let stored = output
        .finish()
        .map_err(|error| (BuildExecutionError::Cas(error), invalidate_cache))?;
    let descriptor = ArtifactDescriptor {
        key,
        storage_class: crate::types::ArtifactStorageClass::Shared,
        kind: spec.kind,
        schema: spec.schema,
        alignment: spec.alignment,
        dependencies: dependency_keys,
        content_hash: stored.content_hash,
        byte_len: stored.byte_len,
    };
    Ok(PreparedJob {
        key,
        descriptor,
        cache_hit: false,
        invalidate_cache,
    })
}

fn run_processor(
    id: JobId,
    processor: &JobProcessor,
    descriptors: &[ArtifactDescriptor],
    store: &RevisionStore,
    max_output_bytes: u64,
    max_materialized_input_bytes: u64,
) -> Result<crate::cas::CasWriter, BuildExecutionError> {
    let failure =
        |source| BuildExecutionError::Graph(BuildError::ProcessorFailed { job: id, source });
    if matches!(processor, JobProcessor::Buffered(_)) {
        admit_materialized_inputs(
            descriptors
                .iter()
                .map(|input| (input.content_hash, input.byte_len)),
            max_materialized_input_bytes,
        )
        .map_err(failure)?;
    }
    let mut inputs = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        let reader = store
            .cas()
            .open_verified(descriptor.content_hash, descriptor.byte_len)?;
        if reader.byte_len() != descriptor.byte_len {
            return Err(BuildExecutionError::InvalidOutput(
                id,
                "dependency CAS length differs from descriptor",
            ));
        }
        inputs.push(ProcessorInput::verified(reader));
    }
    let mut output = store.cas().begin_write(max_output_bytes)?;
    match processor {
        JobProcessor::Buffered(processor) => {
            let payloads =
                materialize_inputs(&mut inputs, max_materialized_input_bytes).map_err(failure)?;
            let refs: Vec<_> = payloads.iter().map(|bytes| bytes.as_ref()).collect();
            let bytes = processor.process(&refs).map_err(failure)?;
            output_limits::check_len(id, bytes.len() as u64, max_output_bytes)?;
            output.write_all(&bytes).map_err(CasError::Io)?;
        }
        JobProcessor::Streaming(processor) => {
            processor
                .execute(&mut inputs, &mut output)
                .map_err(failure)?;
        }
        JobProcessor::Planned(processor) => {
            processor
                .execute(&mut inputs, &mut output)
                .map_err(failure)?;
        }
    }
    // A processor cannot turn an ignored failed write into a valid prefix.
    output.pending_result()?;
    Ok(output)
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

fn validate_dirty(order: &[JobId], dirty: &DirtySet) -> Result<(), BuildExecutionError> {
    let ids: BTreeSet<_> = order.iter().copied().collect();
    if let Some(id) = dirty.jobs().find(|id| !ids.contains(id)) {
        return Err(BuildExecutionError::UnknownDirtyJob(id));
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

    #[test]
    fn buffered_admission_rejects_before_opening_missing_cas_inputs() {
        let root = tempfile::tempdir().unwrap();
        let store = RevisionStore::open(root.path()).unwrap();
        let job = crate::job::BuildJob::new(JobId(7), [], |_: &[&[u8]]| {
            panic!("budget rejection must precede the callback");
        });
        let input =
            ArtifactDescriptor::for_bytes(ArtifactKey([1; 32]), 1, 1, 1, vec![], b"missing input");
        let error = match run_processor(JobId(7), &job.processor(), &[input], &store, 64, 0) {
            Err(error) => error,
            Ok(_) => panic!("over-budget input was accepted"),
        };
        match error {
            BuildExecutionError::Graph(BuildError::ProcessorFailed { job, source }) => {
                assert_eq!(job, JobId(7));
                assert!(source.message().contains("materialization byte bound"));
            }
            other => panic!("expected admission before CAS I/O, got {other:?}"),
        }
        assert_eq!(store.current_revision().unwrap(), None);
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
            importer_version: crate::import::CanonicalPngProcessor::IMPORTER_VERSION,
            canonical_settings: crate::import::CanonicalPngProcessor::canonical_settings(),
            profile_hash: [1; 32],
            toolchain_fingerprint: b"png-0.18".to_vec(),
            environment_fingerprint: b"portable".to_vec(),
        };
        let store = RevisionStore::open(&root).unwrap();
        let mut catalog = crate::processor::InputCatalog::new(&store).unwrap();
        catalog
            .insert_source("source", source_job, &source)
            .unwrap();
        let plan = recipe
            .plan_recorded(
                AssetId(11),
                import_job,
                std::sync::Arc::new(crate::import::CanonicalPngProcessor),
                &catalog,
                vec!["source".into()],
                crate::processor::ProcessLimits::default(),
                crate::processor::ProcessCancellation::default(),
            )
            .unwrap();
        graph.add_job(plan.job).unwrap();
        let mut specs = BTreeMap::new();
        specs.insert(source_job, source.artifact_spec(AssetId(10)));
        specs.insert(import_job, plan.spec);
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
        let image = crate::import::ImageMipArtifact::decode(&bytes).unwrap();
        assert_eq!((image.width, image.height), (1, 1));
        assert_eq!(image.levels.len(), 1);
        assert_eq!(image.levels[0].rgba8, vec![9, 8, 7, 255]);
        assert_eq!(
            execution.artifact_keys[&import_job],
            specs[&import_job]
                .artifact_key(1, &[source.artifact_key()])
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
    fn unrepaired_baseline_corruption_still_blocks_publication() {
        let root = temp_dir("unrepaired-baseline");
        let count = Arc::new(AtomicUsize::new(0));
        let graph = one_job(Arc::clone(&count), b"original");
        let mut specs = BTreeMap::new();
        specs.insert(JobId(1), spec(42, b"settings"));
        let original = run(&graph, &specs, &root, 0);
        let store = RevisionStore::open(&root).unwrap();
        let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();
        let old_hash = original.published.manifest.entries[&AssetId(42)].content_hash;
        fs::write(store.cas().blob_path(old_hash), b"tampered").unwrap();
        let pointer = fs::read(root.join("CURRENT")).unwrap();
        // A changed recipe builds different bytes; it cannot silently abandon
        // an already-published corrupt blob and call the baseline valid.
        let changed = one_job(Arc::clone(&count), b"replacement");
        specs.get_mut(&JobId(1)).unwrap().canonical_settings = b"changed".to_vec();
        let result = BuildExecutor {
            protocol_version: 1,
        }
        .execute(&changed, &specs, &store, &mut ddc, BuildRevision(1));
        assert!(matches!(
            result,
            Err(BuildExecutionError::Publish(PublishError::Cas(
                CasError::Corrupt { .. }
            )))
        ));
        assert_eq!(fs::read(root.join("CURRENT")).unwrap(), pointer);
        assert!(!root.join("revisions/1.manifest").exists());
        assert_eq!(count.load(Ordering::Relaxed), 2);
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
