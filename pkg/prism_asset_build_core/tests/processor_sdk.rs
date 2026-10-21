//! The recorded SDK runs through the real DAG/DDC/CAS/publication boundary.
use prism_asset_build_core::processor::{
    prepare_process, AnalyzeContext, ArtifactType, ArtifactWriter, ExecuteContext, InputCatalog,
    PreparedProcess, ProcessCancellation, ProcessLimits, ProcessPlan, ProcessReport,
    ProcessRequest, Processor, ProcessorCategory, ProcessorDescriptor,
};
use prism_asset_build_core::{
    ArtifactDescriptor, ArtifactSpec, AssetId, BuildExecution, BuildExecutionError, BuildExecutor,
    BuildGraph, BuildResources, BuildRevision, ContentHash, DerivedDataCache, JobId,
    ProcessorError, ProcessorIdentity, RevisionStore, SourceSnapshot,
};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

const SOURCE: ArtifactType = ArtifactType {
    kind: prism_asset_build_core::source::SOURCE_ARTIFACT_KIND,
    schema: prism_asset_build_core::source::SOURCE_ARTIFACT_SCHEMA,
};
const OUTPUT: ArtifactType = ArtifactType {
    kind: 123,
    schema: 1,
};
const OUTPUT_JOB: JobId = JobId(100);

#[derive(Clone, Copy, Debug)]
enum ReadFailure {
    Exact,
    Utf8,
    ToEnd,
}

fn ignore_read_failure(input: &mut impl Read, failure: ReadFailure) {
    let error = match failure {
        ReadFailure::Exact => input.read_exact(&mut [0; 2]).unwrap_err(),
        ReadFailure::Utf8 => input.read_to_string(&mut String::new()).unwrap_err(),
        ReadFailure::ToEnd => input.read_to_end(&mut Vec::new()).unwrap_err(),
    };
    assert_eq!(
        error.kind(),
        match failure {
            ReadFailure::Exact => std::io::ErrorKind::UnexpectedEof,
            ReadFailure::Utf8 => std::io::ErrorKind::InvalidData,
            ReadFailure::ToEnd => std::io::ErrorKind::Other,
        }
    );
    // The plugin deliberately ignores the confirmed error and returns success.
}

struct FailedFormat;

impl std::fmt::Display for FailedFormat {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Err(std::fmt::Error)
    }
}

#[derive(Clone, Copy)]
enum Behavior {
    Good,
    IgnoredUndeclared,
    IgnoredWriteError,
    BadReport,
    Panic,
    Cancel,
    NonDeterministic,
    AnalyzePanic,
    IgnoredAnalysisError,
    IgnoredAnalysisRead(ReadFailure),
    IgnoredExecuteRead(ReadFailure),
    IgnoredFormatFailure,
}

struct Plugin {
    descriptor: ProcessorDescriptor,
    behavior: Behavior,
    calls: Arc<AtomicUsize>,
    cancelled: ProcessCancellation,
    cacheable: bool,
}

impl Plugin {
    fn new(behavior: Behavior) -> Self {
        Self {
            descriptor: ProcessorDescriptor {
                identity: ProcessorIdentity::new(789, 1),
                name: "recorded test".to_owned(),
                category: ProcessorCategory::Compiler,
                input_types: vec![SOURCE],
                output_types: vec![OUTPUT],
                settings_schema: 456,
                deterministic: true,
                supports_cancellation: true,
            },
            behavior,
            calls: Arc::new(AtomicUsize::new(0)),
            cancelled: ProcessCancellation::default(),
            cacheable: true,
        }
    }
}

impl Processor for Plugin {
    fn descriptor(&self) -> &ProcessorDescriptor {
        &self.descriptor
    }

    fn analyze(
        &self,
        ctx: &mut AnalyzeContext<'_>,
        _: &ProcessRequest,
    ) -> Result<ProcessPlan, ProcessorError> {
        if matches!(self.behavior, Behavior::AnalyzePanic) {
            panic!("bad analysis");
        }
        if matches!(self.behavior, Behavior::IgnoredAnalysisError) {
            let _ = ctx.declare_input("missing.bin");
        }
        let mut dependency = String::new();
        ctx.open_input("root.bin")?
            .read_to_string(&mut dependency)
            .map_err(|error| ProcessorError::new(error.to_string()))?;
        ctx.declare_input(&dependency)?;
        if let Behavior::IgnoredAnalysisRead(failure) = self.behavior {
            ignore_read_failure(&mut ctx.open_input("dep.bin")?, failure);
        }
        Ok(ProcessPlan {
            output: OUTPUT,
            alignment: 16,
            canonical_settings: b"frozen".to_vec(),
            resources: BuildResources {
                memory_bytes: 65536,
                ..BuildResources::default()
            },
            cacheable: self.cacheable,
        })
    }

    fn execute(
        &self,
        ctx: &mut ExecuteContext<'_, '_>,
        _: &ProcessPlan,
        output: &mut ArtifactWriter<'_>,
    ) -> Result<ProcessReport, ProcessorError> {
        let run = self.calls.fetch_add(1, Ordering::SeqCst);
        let io_error = |error: std::io::Error| ProcessorError::new(error.to_string());
        output.write_all(b"private:").map_err(io_error)?;
        match self.behavior {
            Behavior::IgnoredUndeclared => {
                let _ = ctx.input("undeclared.bin");
            }
            Behavior::IgnoredWriteError => {
                let _ = output.write_all(&[0; 256]);
            }
            Behavior::Panic => panic!("bad execute"),
            Behavior::Cancel => self.cancelled.cancel(),
            Behavior::NonDeterministic => {
                output.write_all(&run.to_le_bytes()).map_err(io_error)?;
            }
            Behavior::IgnoredExecuteRead(failure) => {
                ignore_read_failure(&mut ctx.input("dep.bin")?, failure);
                return Ok(ProcessReport {
                    output_bytes: output.bytes_written(),
                });
            }
            Behavior::IgnoredFormatFailure => {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    output.write_fmt(format_args!("{}", FailedFormat))
                }));
                assert!(!matches!(result, Ok(Ok(()))));
                return Ok(ProcessReport {
                    output_bytes: output.bytes_written(),
                });
            }
            _ => {}
        }
        for name in ctx.input_names().to_vec() {
            let mut input = ctx.input(&name)?;
            std::io::copy(&mut input, output).map_err(io_error)?;
        }
        Ok(ProcessReport {
            output_bytes: output.bytes_written()
                + u64::from(matches!(self.behavior, Behavior::BadReport)),
        })
    }
}

struct Fixture {
    _root: tempfile::TempDir,
    store: RevisionStore,
    ddc: DerivedDataCache,
    sources: Vec<(JobId, SourceSnapshot)>,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let store = RevisionStore::open(root.path().join("store")).unwrap();
        let ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
        let sources = vec![
            (
                JobId(1),
                SourceSnapshot::new("root.bin", b"dep.bin".to_vec()).unwrap(),
            ),
            (
                JobId(2),
                SourceSnapshot::new("dep.bin", b"payload".to_vec()).unwrap(),
            ),
        ];
        Self {
            _root: root,
            store,
            ddc,
            sources,
        }
    }

    fn catalog(&self) -> InputCatalog<'_> {
        let mut catalog = InputCatalog::new(&self.store).unwrap();
        for (job, source) in &self.sources {
            self.store.cas().put(source.bytes()).unwrap();
            let spec = source.artifact_spec(AssetId(job.0));
            let descriptor = ArtifactDescriptor::for_bytes(
                spec.artifact_key(1, &[]).unwrap(),
                spec.kind,
                spec.schema,
                spec.alignment,
                vec![],
                source.bytes(),
            );
            catalog
                .insert(source.logical_path(), *job, descriptor)
                .unwrap();
        }
        catalog
    }

    fn prepare(
        &self,
        plugin: Arc<Plugin>,
        limits: ProcessLimits,
    ) -> Result<PreparedProcess, ProcessorError> {
        let token = plugin.cancelled.clone();
        prepare_process(plugin, &self.catalog(), request(), limits, token)
    }

    fn run(
        &mut self,
        prepared: PreparedProcess,
        revision: u64,
        verified: bool,
    ) -> Result<BuildExecution, BuildExecutionError> {
        let mut graph = BuildGraph::new();
        let mut specs = BTreeMap::new();
        for (job, source) in &self.sources {
            graph.add_job(source.build_job(*job)).unwrap();
            specs.insert(*job, source.artifact_spec(AssetId(job.0)));
        }
        specs.insert(OUTPUT_JOB, prepared.spec);
        graph.add_job(prepared.job).unwrap();
        let executor = BuildExecutor {
            protocol_version: 1,
        };
        if verified {
            executor.execute_verified_with_limit(
                &graph,
                &specs,
                &self.store,
                &mut self.ddc,
                BuildRevision(revision),
                2,
            )
        } else {
            executor.execute(
                &graph,
                &specs,
                &self.store,
                &mut self.ddc,
                BuildRevision(revision),
            )
        }
    }

    fn assert_unpublished(&self) {
        assert_eq!(self.store.current_revision().unwrap(), None);
        assert!(!self
            .store
            .cas()
            .blob_path(ContentHash::digest(b"private:"))
            .exists());
        assert!(!self
            .store
            .cas()
            .blob_path(ContentHash::digest(b"private:payloaddep.bin"))
            .exists());
    }
}

fn request() -> ProcessRequest {
    ProcessRequest {
        job: OUTPUT_JOB,
        asset: AssetId(100),
        primary_inputs: vec!["root.bin".into()],
        canonical_settings: b"requested".to_vec(),
        profile_hash: [7; 32],
        toolchain_fingerprint: b"toolchain-v1".to_vec(),
        environment_fingerprint: vec![],
    }
}

#[test]
fn dynamic_dependencies_stream_through_verified_execution_then_reuse_cache() {
    let mut fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::Good));
    let prepared = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    assert_eq!(prepared.job.dependencies(), &[JobId(2), JobId(1)]);
    assert_eq!(prepared.job.resources().memory_bytes, 65536);
    let result = fixture.run(prepared, 0, true).unwrap();
    let descriptor = &result.published.manifest.entries[&AssetId(100)];
    assert_eq!(
        fixture.store.cas().read(descriptor.content_hash).unwrap(),
        b"private:payloaddep.bin"
    );
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 2);
    assert_eq!(result.output_memory.peak_retained_bytes, 0);
    let prepared = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    let second = fixture.run(prepared, 1, false).unwrap();
    assert!(second.cache_hits.contains(&OUTPUT_JOB));
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn named_aliases_keep_independent_streams_and_publish_unique_dependencies() {
    let mut fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::Good));
    let prepare = |fixture: &Fixture| {
        let mut catalog = fixture.catalog();
        let (job, source) = &fixture.sources[1];
        let spec = source.artifact_spec(AssetId(job.0));
        let descriptor = ArtifactDescriptor::for_bytes(
            spec.artifact_key(1, &[]).unwrap(),
            spec.kind,
            spec.schema,
            spec.alignment,
            vec![],
            source.bytes(),
        );
        catalog.insert("alias.bin", *job, descriptor).unwrap();
        let mut request = request();
        request.primary_inputs.push("alias.bin".into());
        prepare_process(
            plugin.clone(),
            &catalog,
            request,
            ProcessLimits::default(),
            plugin.cancelled.clone(),
        )
        .unwrap()
    };
    let prepared = prepare(&fixture);
    assert_eq!(prepared.job.dependencies(), &[JobId(2), JobId(2), JobId(1)]);
    let result = fixture.run(prepared, 0, true).unwrap();
    let entries = &result.published.manifest.entries;
    let output = &entries[&AssetId(100)];
    assert_eq!(
        output.dependencies,
        [entries[&AssetId(2)].key, entries[&AssetId(1)].key]
    );
    assert_eq!(
        fixture.store.cas().read(output.content_hash).unwrap(),
        b"private:payloadpayloaddep.bin"
    );
    let prepared = prepare(&fixture);
    let warm = fixture.run(prepared, 1, false).unwrap();
    assert!(warm.cache_hits.contains(&OUTPUT_JOB));
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn frozen_contract_rejects_spec_tampering_even_with_a_cache_hit() {
    let mut fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::Good));
    let first = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    fixture.run(first, 0, false).unwrap();
    let mut next = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    next.spec = ArtifactSpec {
        schema: 99,
        ..next.spec
    };
    assert!(fixture
        .run(next, 1, false)
        .unwrap_err()
        .to_string()
        .contains("frozen processor plan"));
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(BuildRevision(0))
    );
}

#[test]
fn graph_inputs_changed_after_analysis_are_rejected_before_plugin_execution() {
    let mut fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::Good));
    let prepared = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    fixture.sources[1].1 = SourceSnapshot::new("dep.bin", b"changed".to_vec()).unwrap();
    assert!(fixture
        .run(prepared, 0, false)
        .unwrap_err()
        .to_string()
        .contains("execution inputs differ"));
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 0);
    fixture.assert_unpublished();
}

#[test]
fn frozen_resource_admission_cannot_be_reduced_by_the_graph_caller() {
    let mut fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::Good));
    let mut prepared = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    prepared.job = prepared.job.with_resources(BuildResources::default());
    assert!(fixture
        .run(prepared, 0, false)
        .unwrap_err()
        .to_string()
        .contains("job resources differ"));
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 0);
    assert!(fixture.ddc.is_empty());
    fixture.assert_unpublished();
}

#[test]
fn warm_cache_cannot_bypass_the_frozen_output_limit() {
    let mut fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::Good));
    let prepared = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    fixture.run(prepared, 0, false).unwrap();
    let limits = ProcessLimits {
        max_output_bytes: 10,
        ..ProcessLimits::default()
    };
    let mut prepared = fixture.prepare(plugin.clone(), limits).unwrap();
    prepared.job = prepared.job.with_max_output_bytes(u64::MAX);
    assert_eq!(prepared.job.max_output_bytes(), 10);
    assert!(fixture.run(prepared, 1, false).is_err());
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(BuildRevision(0))
    );
}

#[test]
fn dependency_content_and_request_settings_both_enter_the_artifact_key() {
    let fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::Good));
    let first = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    let catalog = fixture.catalog();
    let mut changed_request = request();
    changed_request.canonical_settings.push(0);
    let second = prepare_process(
        plugin.clone(),
        &catalog,
        changed_request,
        ProcessLimits::default(),
        plugin.cancelled.clone(),
    )
    .unwrap();
    assert_ne!(
        first.spec.artifact_key(1, &[]).unwrap(),
        second.spec.artifact_key(1, &[]).unwrap()
    );
    drop(catalog);
    let mut fixture = fixture;
    fixture.sources[1].1 = SourceSnapshot::new("dep.bin", b"new payload".to_vec()).unwrap();
    let third = fixture.prepare(plugin, ProcessLimits::default()).unwrap();
    assert_ne!(
        first.spec.artifact_key(1, &[]).unwrap(),
        third.spec.artifact_key(1, &[]).unwrap()
    );
}

#[test]
fn ignored_undeclared_access_write_error_and_bad_report_cannot_publish() {
    for behavior in [
        Behavior::IgnoredUndeclared,
        Behavior::IgnoredWriteError,
        Behavior::BadReport,
        Behavior::Panic,
    ] {
        let mut fixture = Fixture::new();
        let plugin = Arc::new(Plugin::new(behavior));
        let limits = ProcessLimits {
            max_output_bytes: 64,
            ..ProcessLimits::default()
        };
        let prepared = fixture.prepare(plugin, limits).unwrap();
        assert!(fixture.run(prepared, 0, false).is_err());
        fixture.assert_unpublished();
    }
}

#[test]
fn ignored_analysis_read_exact_eof_rejects_the_plan() {
    assert_ignored_analysis_read_rejected(ReadFailure::Exact);
}

#[test]
fn ignored_analysis_read_to_string_invalid_utf8_rejects_the_plan() {
    assert_ignored_analysis_read_rejected(ReadFailure::Utf8);
}

#[test]
fn ignored_analysis_read_to_end_budget_error_rejects_the_plan() {
    assert_ignored_analysis_read_rejected(ReadFailure::ToEnd);
}

fn assert_ignored_analysis_read_rejected(failure: ReadFailure) {
    let mut fixture = Fixture::new();
    fixture.sources[1].1 = SourceSnapshot::new("dep.bin", vec![0xff]).unwrap();
    let plugin = Arc::new(Plugin::new(Behavior::IgnoredAnalysisRead(failure)));
    let mut limits = ProcessLimits::default();
    if matches!(failure, ReadFailure::ToEnd) {
        // root verification + root reads + dep verification; no dep read left.
        limits.max_analysis_bytes = 15;
    }
    let error = fixture.prepare(plugin.clone(), limits).err().unwrap();
    assert!(
        error.to_string().contains("analysis input access failed"),
        "{failure:?}: {error}"
    );
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 0);
    fixture.assert_unpublished();
}

#[test]
fn ignored_execute_read_exact_eof_cannot_publish() {
    assert_ignored_execute_failure_rejected(Behavior::IgnoredExecuteRead(ReadFailure::Exact));
}

#[test]
fn ignored_execute_read_to_string_invalid_utf8_cannot_publish() {
    assert_ignored_execute_failure_rejected(Behavior::IgnoredExecuteRead(ReadFailure::Utf8));
}

#[test]
fn caught_write_fmt_failure_cannot_publish() {
    assert_ignored_execute_failure_rejected(Behavior::IgnoredFormatFailure);
}

fn assert_ignored_execute_failure_rejected(behavior: Behavior) {
    let mut fixture = Fixture::new();
    fixture.sources[1].1 = SourceSnapshot::new("dep.bin", vec![0xff]).unwrap();
    let plugin = Arc::new(Plugin::new(behavior));
    let prepared = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    let key = prepared
        .spec
        .artifact_key(
            1,
            &prepared
                .job
                .dependencies()
                .iter()
                .map(|id| {
                    let (_, source) = fixture.sources.iter().find(|(job, _)| job == id).unwrap();
                    source
                        .artifact_spec(AssetId(id.0))
                        .artifact_key(1, &[])
                        .unwrap()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let error = fixture.run(prepared, 0, false).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("execution input or output access failed"),
        "{error}"
    );
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 1);
    assert!(fixture.ddc.get(key).is_none());
    fixture.assert_unpublished();
}

#[test]
fn cancellation_during_execution_and_before_cache_reuse_prevents_publication() {
    let mut fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::Cancel));
    let prepared = fixture.prepare(plugin, ProcessLimits::default()).unwrap();
    assert!(fixture
        .run(prepared, 0, false)
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    fixture.assert_unpublished();

    let mut fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::Good));
    let prepared = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    fixture.run(prepared, 0, false).unwrap();
    let prepared = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    plugin.cancelled.cancel();
    assert!(fixture
        .run(prepared, 1, false)
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(BuildRevision(0))
    );
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn noncacheable_plans_never_read_or_write_ddc_for_the_output() {
    let mut fixture = Fixture::new();
    let mut plugin = Plugin::new(Behavior::Good);
    plugin.cacheable = false;
    let plugin = Arc::new(plugin);
    for revision in 0..=1 {
        let prepared = fixture
            .prepare(plugin.clone(), ProcessLimits::default())
            .unwrap();
        let result = fixture.run(prepared, revision, false).unwrap();
        assert!(!result.cache_hits.contains(&OUTPUT_JOB));
        assert!(fixture.ddc.get(result.artifact_keys[&OUTPUT_JOB]).is_none());
    }
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn noncacheable_plan_also_bypasses_an_existing_cache_entry() {
    let mut fixture = Fixture::new();
    let cached = Arc::new(Plugin::new(Behavior::Good));
    let prepared = fixture.prepare(cached, ProcessLimits::default()).unwrap();
    fixture.run(prepared, 0, false).unwrap();
    let mut fresh = Plugin::new(Behavior::Good);
    fresh.cacheable = false;
    let fresh = Arc::new(fresh);
    let prepared = fixture
        .prepare(fresh.clone(), ProcessLimits::default())
        .unwrap();
    let result = fixture.run(prepared, 1, false).unwrap();
    assert!(!result.cache_hits.contains(&OUTPUT_JOB));
    assert_eq!(fresh.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn analysis_errors_panics_type_capabilities_and_budgets_reject_the_plan() {
    let fixture = Fixture::new();
    for behavior in [Behavior::AnalyzePanic, Behavior::IgnoredAnalysisError] {
        assert!(fixture
            .prepare(Arc::new(Plugin::new(behavior)), ProcessLimits::default())
            .is_err());
    }
    let mut plugin = Plugin::new(Behavior::Good);
    plugin.descriptor.input_types.clear();
    assert!(fixture
        .prepare(Arc::new(plugin), ProcessLimits::default())
        .err()
        .unwrap()
        .to_string()
        .contains("capabilities"));
    for limits in [
        ProcessLimits {
            max_inputs: 1,
            ..ProcessLimits::default()
        },
        ProcessLimits {
            max_analysis_bytes: 13,
            ..ProcessLimits::default()
        },
        ProcessLimits {
            max_settings_bytes: 1,
            ..ProcessLimits::default()
        },
    ] {
        assert!(fixture
            .prepare(Arc::new(Plugin::new(Behavior::Good)), limits)
            .is_err());
    }
    let exact = ProcessLimits {
        max_inputs: 2,
        max_analysis_bytes: 14,
        ..ProcessLimits::default()
    };
    assert!(fixture
        .prepare(Arc::new(Plugin::new(Behavior::Good)), exact)
        .is_ok());
    fixture.assert_unpublished();
}

#[test]
fn deterministic_double_run_discards_both_sdk_outputs_on_mismatch() {
    let mut fixture = Fixture::new();
    let plugin = Arc::new(Plugin::new(Behavior::NonDeterministic));
    let prepared = fixture
        .prepare(plugin.clone(), ProcessLimits::default())
        .unwrap();
    assert!(matches!(
        fixture.run(prepared, 0, true),
        Err(BuildExecutionError::NonDeterministic(_))
    ));
    assert_eq!(plugin.calls.load(Ordering::SeqCst), 2);
    fixture.assert_unpublished();
    for run in 0usize..2 {
        let mut bytes = b"private:".to_vec();
        bytes.extend_from_slice(&run.to_le_bytes());
        bytes.extend_from_slice(b"payloaddep.bin");
        assert!(!fixture
            .store
            .cas()
            .blob_path(ContentHash::digest(&bytes))
            .exists());
    }
}
