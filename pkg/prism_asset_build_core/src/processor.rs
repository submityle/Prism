//! Trusted processor SDK: recorded analysis, frozen inputs, and transactional
//! execution through the build executor. Ambient process I/O is not sandboxed.

use crate::cas::VerifiedCasReader;
use crate::job::{BuildJob, JobId, ProcessorError, ProcessorInput};
use crate::{
    ArtifactDescriptor, ArtifactSpec, AssetId, BuildOutputLease, BuildResources, ProcessorIdentity,
    RevisionStore,
};
use std::collections::BTreeMap;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

/// A kind and schema accepted or produced by a processor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactType {
    /// Stable protocol kind ID.
    pub kind: u128,
    /// Explicit wire schema version.
    pub schema: u32,
}

/// Role of a processor in the content pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessorCategory {
    /// Source format conversion.
    Importer,
    /// Derived content generation.
    Baker,
    /// Target artifact compilation.
    Compiler,
    /// Content validation.
    Validator,
    /// Distribution layout generation.
    Packager,
}

/// Semantic identity and capabilities checked before invoking plugin code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessorDescriptor {
    /// Stable implementation identity; semantic changes require a version bump.
    pub identity: ProcessorIdentity,
    /// Human-readable diagnostic name.
    pub name: String,
    /// Content pipeline role.
    pub category: ProcessorCategory,
    /// Types the analysis context may declare or open.
    pub input_types: Vec<ArtifactType>,
    /// Permitted output contracts.
    pub output_types: Vec<ArtifactType>,
    /// Stable schema ID for the canonical request settings.
    pub settings_schema: u128,
    /// Shared artifacts require deterministic processors.
    pub deterministic: bool,
    /// Whether plugin CPU work cooperatively polls cancellation between I/O.
    /// SDK I/O checks cancellation regardless of this declaration.
    pub supports_cancellation: bool,
}

/// Explicit immutable request. Environment and toolchain inputs must be
/// captured here; the SDK never reads process environment variables.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessRequest {
    /// Identity of the node to add to the DAG.
    pub job: JobId,
    /// Logical output asset.
    pub asset: AssetId,
    /// Initial dependencies; the analyzer may record additional inputs.
    pub primary_inputs: Vec<String>,
    /// Canonically encoded caller settings, included verbatim in key material.
    pub canonical_settings: Vec<u8>,
    /// Explicit profile projection digest.
    pub profile_hash: [u8; 32],
    /// Explicit compiler and tool versions.
    pub toolchain_fingerprint: Vec<u8>,
    /// Explicit output-affecting environment values.
    pub environment_fingerprint: Vec<u8>,
}

/// Analysis result. Declared inputs are recorded by AnalyzeContext, rather
/// than accepting an unverified dependency list returned by the plugin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessPlan {
    /// Selected output kind/schema, checked against the descriptor.
    pub output: ArtifactType,
    /// Required nonzero power-of-two alignment.
    pub alignment: u32,
    /// Immutable execution configuration produced by analysis.
    pub canonical_settings: Vec<u8>,
    /// Concurrent scheduler admission requirements.
    pub resources: BuildResources,
    /// Whether this job may read or write shared DDC entries.
    pub cacheable: bool,
}

/// Byte and entry limits enforced independently of plugin resource estimates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessLimits {
    /// Maximum number of distinct named dependencies.
    pub max_inputs: usize,
    /// Aggregate analysis hash-verification and read bytes, including rereads.
    pub max_analysis_bytes: u64,
    /// Maximum length of each explicit settings or fingerprint blob.
    pub max_settings_bytes: usize,
    /// Maximum encoded artifact payload length.
    pub max_output_bytes: u64,
}

impl Default for ProcessLimits {
    fn default() -> Self {
        Self {
            max_inputs: 4096,
            max_analysis_bytes: 64 * 1024 * 1024,
            max_settings_bytes: 1024 * 1024,
            max_output_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Cooperative cancellation checked before cache reuse, on SDK I/O, and at
/// output acceptance. It cannot interrupt arbitrary trusted plugin CPU code.
#[derive(Clone, Debug, Default)]
pub struct ProcessCancellation(Arc<AtomicBool>);

impl ProcessCancellation {
    /// Signal cooperative cancellation to the whole prepared process.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Whether cancellation has been signalled.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    /// Fail when cancellation has been signalled.
    pub fn check(&self) -> Result<(), ProcessorError> {
        if self.is_cancelled() {
            Err(error("processor cancelled"))
        } else {
            Ok(())
        }
    }
}

/// The SDK is for trusted in-process plugins. Invalid API accesses are sticky
/// errors, so ignoring an undeclared-input or writer error cannot publish output.
pub trait Processor: Send + Sync {
    /// Stable processor identity and capabilities.
    fn descriptor(&self) -> &ProcessorDescriptor;
    /// Discover inputs and select an output contract without publishing bytes.
    fn analyze(
        &self,
        ctx: &mut AnalyzeContext<'_>,
        request: &ProcessRequest,
    ) -> Result<ProcessPlan, ProcessorError>;
    /// Produce the one private artifact using only frozen named inputs.
    fn execute(
        &self,
        ctx: &mut ExecuteContext<'_, '_>,
        plan: &ProcessPlan,
        output: &mut ArtifactWriter<'_>,
    ) -> Result<ProcessReport, ProcessorError>;
}

/// Execution accounting checked against the actual output writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessReport {
    /// Number of payload bytes accepted by the writer.
    pub output_bytes: u64,
}

#[derive(Clone)]
struct DeclaredInput {
    job: JobId,
    descriptor: ArtifactDescriptor,
}

/// Caller-owned immutable artifact catalog. Its store lease protects analysis
/// inputs from cooperating GC until the catalog is dropped. Graph execution
/// acquires its own lease; missing inputs between the two fail closed.
pub struct InputCatalog<'a> {
    store: &'a RevisionStore,
    entries: BTreeMap<String, DeclaredInput>,
    descriptors_by_job: BTreeMap<JobId, ArtifactDescriptor>,
    _lease: BuildOutputLease,
}

impl<'a> InputCatalog<'a> {
    /// Create an empty catalog while holding a cooperating store-GC lease.
    pub fn new(store: &'a RevisionStore) -> Result<Self, ProcessorError> {
        Ok(Self {
            store,
            entries: BTreeMap::new(),
            descriptors_by_job: BTreeMap::new(),
            _lease: store.protect_build_outputs().map_err(display_error)?,
        })
    }

    /// Persists an immutable source under this catalog's GC lease and binds
    /// its exact root-job descriptor. No build revision is published; failed
    /// planning may leave an unreferenced, valid CAS object for explicit GC.
    pub fn insert_source(
        &mut self,
        name: &str,
        job: JobId,
        source: &crate::SourceSnapshot,
    ) -> Result<(), ProcessorError> {
        let descriptor = ArtifactDescriptor {
            key: source.artifact_key(),
            kind: crate::source::SOURCE_ARTIFACT_KIND,
            schema: crate::source::SOURCE_ARTIFACT_SCHEMA,
            storage_class: crate::ArtifactStorageClass::Shared,
            content_hash: source.content_hash(),
            byte_len: source.bytes().len() as u64,
            alignment: 1,
            dependencies: Vec::new(),
        };
        if !crate::source::is_canonical_path(name)
            || self.entries.contains_key(name)
            || self
                .descriptors_by_job
                .get(&job)
                .is_some_and(|existing| existing != &descriptor)
        {
            return Err(error("invalid or conflicting source input binding"));
        }
        self.store
            .cas()
            .put(source.bytes())
            .map_err(display_error)?;
        self.insert(name, job, descriptor)
    }

    /// Bind a canonical input name to an immutable upstream job descriptor.
    pub fn insert(
        &mut self,
        name: &str,
        job: JobId,
        descriptor: ArtifactDescriptor,
    ) -> Result<(), ProcessorError> {
        if !crate::source::is_canonical_path(name) {
            return Err(error(
                "processor input name is not a canonical relative path",
            ));
        }
        descriptor.validate().map_err(error)?;
        if !descriptor.storage_class.is_shared() {
            return Err(error("processor input must be a shared artifact"));
        }
        if self.entries.contains_key(name) {
            return Err(error("duplicate processor input name"));
        }
        if self
            .descriptors_by_job
            .get(&job)
            .is_some_and(|existing| existing != &descriptor)
        {
            return Err(error("one input job has contradictory descriptors"));
        }
        self.descriptors_by_job
            .entry(job)
            .or_insert_with(|| descriptor.clone());
        self.entries
            .insert(name.to_owned(), DeclaredInput { job, descriptor });
        Ok(())
    }
}

/// Records every primary or dynamically discovered dependency in canonical
/// name order. Only declared artifact types can be opened during analysis.
pub struct AnalyzeContext<'a> {
    store: &'a RevisionStore,
    catalog: &'a BTreeMap<String, DeclaredInput>,
    permitted: &'a [ArtifactType],
    declared: BTreeMap<String, DeclaredInput>,
    limits: ProcessLimits,
    remaining: Arc<AtomicU64>,
    cancelled: ProcessCancellation,
    failed: Arc<AtomicBool>,
}

impl AnalyzeContext<'_> {
    /// Record an input without reading its payload.
    pub fn declare_input(&mut self, name: &str) -> Result<(), ProcessorError> {
        let result = (|| {
            self.cancelled.check()?;
            let input = self
                .catalog
                .get(name)
                .ok_or_else(|| error("analysis input is absent"))?;
            let input_type = ArtifactType {
                kind: input.descriptor.kind,
                schema: input.descriptor.schema,
            };
            if !self.permitted.contains(&input_type) {
                return Err(error(
                    "analysis input type is outside processor capabilities",
                ));
            }
            if !self.declared.contains_key(name) && self.declared.len() >= self.limits.max_inputs {
                return Err(error("processor declared input count exceeds limit"));
            }
            self.declared.insert(name.to_owned(), input.clone());
            Ok(())
        })();
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }

    /// Opening an input records the dependency even when only its header is
    /// read. Hash verification is admitted against the analysis byte budget.
    /// The reader borrows this analysis scope and cannot be stored in a plugin
    /// for later execution.
    ///
    /// ```compile_fail
    /// use prism_asset_build_core::processor::{AnalyzeContext, AnalysisInput};
    /// fn escape(ctx: &mut AnalyzeContext<'_>) -> AnalysisInput<'static> {
    ///     ctx.open_input("input.bin").unwrap()
    /// }
    /// ```
    pub fn open_input(&mut self, name: &str) -> Result<AnalysisInput<'_>, ProcessorError> {
        self.declare_input(name)?;
        let descriptor = &self.declared[name].descriptor;
        let result = (|| {
            charge(&self.remaining, descriptor.byte_len)?;
            let reader = self
                .store
                .cas()
                .open_verified(descriptor.content_hash, descriptor.byte_len)
                .map_err(display_error)?;
            if reader.byte_len() != descriptor.byte_len {
                return Err(error(
                    "analysis input length differs from frozen descriptor",
                ));
            }
            Ok(AnalysisInput {
                reader,
                remaining: &self.remaining,
                cancelled: &self.cancelled,
                failed: &self.failed,
            })
        })();
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
}

/// Seekable verified analysis stream. Verification and subsequent byte reads
/// share one aggregate budget; repeated seeks cannot bypass the read limit.
pub struct AnalysisInput<'a> {
    reader: VerifiedCasReader,
    remaining: &'a AtomicU64,
    cancelled: &'a ProcessCancellation,
    failed: &'a AtomicBool,
}

impl Read for AnalysisInput<'_> {
    fn read_exact(&mut self, bytes: &mut [u8]) -> io::Result<()> {
        let failed = self.failed;
        sticky_io(failed, || PrimitiveIo(self).read_exact(bytes))
    }

    fn read_to_end(&mut self, bytes: &mut Vec<u8>) -> io::Result<usize> {
        let failed = self.failed;
        sticky_io(failed, || PrimitiveIo(self).read_to_end(bytes))
    }

    fn read_to_string(&mut self, text: &mut String) -> io::Result<usize> {
        let failed = self.failed;
        sticky_io(failed, || PrimitiveIo(self).read_to_string(text))
    }

    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let result = (|| {
            self.cancelled.check().map_err(io::Error::other)?;
            let left = self
                .reader
                .byte_len()
                .saturating_sub(self.reader.stream_position()?);
            let count = (bytes.len() as u64).min(left) as usize;
            charge(&self.remaining, count as u64).map_err(io::Error::other)?;
            self.reader.read(&mut bytes[..count])
        })();
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
}

impl Seek for AnalysisInput<'_> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let result = self
            .cancelled
            .check()
            .map_err(io::Error::other)
            .and_then(|()| self.reader.seek(from));
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
}

/// Frozen input access for execute. No catalog or store is exposed here.
pub struct ExecuteContext<'a, 'input> {
    inputs: &'a mut [ProcessorInput<'input>],
    names: &'a [String],
    cancelled: &'a ProcessCancellation,
    failed: &'a AtomicBool,
}

impl<'a, 'input> ExecuteContext<'a, 'input> {
    /// Borrow a declared stream at its current position. Reopening does not
    /// rewind it; processors can explicitly seek within the verified bounds.
    pub fn input(&mut self, name: &str) -> Result<ExecuteInput<'_, 'input>, ProcessorError> {
        let result = (|| {
            self.cancelled.check()?;
            let slot = self
                .names
                .binary_search_by(|item| item.as_str().cmp(name))
                .map_err(|_| error("execute attempted an undeclared input"))?;
            Ok(slot)
        })();
        match result {
            Ok(slot) => Ok(ExecuteInput {
                reader: &mut self.inputs[slot],
                cancelled: self.cancelled,
                failed: self.failed,
            }),
            Err(error) => {
                self.failed.store(true, Ordering::Release);
                Err(error)
            }
        }
    }

    /// Cooperative cancellation point for CPU work between I/O operations.
    pub fn check_cancelled(&self) -> Result<(), ProcessorError> {
        self.cancelled.check()
    }
    /// Frozen dependency names in canonical ascending order.
    pub fn input_names(&self) -> &[String] {
        self.names
    }
}

/// Borrowed execution stream whose I/O failures poison the candidate output.
pub struct ExecuteInput<'a, 'input> {
    reader: &'a mut ProcessorInput<'input>,
    cancelled: &'a ProcessCancellation,
    failed: &'a AtomicBool,
}

impl Read for ExecuteInput<'_, '_> {
    fn read_exact(&mut self, bytes: &mut [u8]) -> io::Result<()> {
        let failed = self.failed;
        sticky_io(failed, || PrimitiveIo(self).read_exact(bytes))
    }

    fn read_to_end(&mut self, bytes: &mut Vec<u8>) -> io::Result<usize> {
        let failed = self.failed;
        sticky_io(failed, || PrimitiveIo(self).read_to_end(bytes))
    }

    fn read_to_string(&mut self, text: &mut String) -> io::Result<usize> {
        let failed = self.failed;
        sticky_io(failed, || PrimitiveIo(self).read_to_string(text))
    }

    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let result = self
            .cancelled
            .check()
            .map_err(io::Error::other)
            .and_then(|()| self.reader.read(bytes));
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
}

impl Seek for ExecuteInput<'_, '_> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let result = self
            .cancelled
            .check()
            .map_err(io::Error::other)
            .and_then(|()| self.reader.seek(from));
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
}

/// Private bounded artifact stream. Finalization and publication remain owned
/// by BuildExecutor; a plugin cannot commit a partial output itself.
pub struct ArtifactWriter<'a> {
    output: &'a mut dyn Write,
    cancelled: &'a ProcessCancellation,
    failed: &'a AtomicBool,
    byte_len: u64,
    max_bytes: u64,
}

impl ArtifactWriter<'_> {
    /// Actual accepted byte count, available for the final execution report.
    pub fn bytes_written(&self) -> u64 {
        self.byte_len
    }
}

impl Write for ArtifactWriter<'_> {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        let failed = self.failed;
        sticky_io(failed, || PrimitiveIo(self).write_all(bytes))
    }

    fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) -> io::Result<()> {
        let failed = self.failed;
        sticky_io(failed, || PrimitiveIo(self).write_fmt(args))
    }

    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let result = (|| {
            if self.failed.load(Ordering::Acquire) {
                return Err(io::Error::other("processor stream failed"));
            }
            self.cancelled.check().map_err(io::Error::other)?;
            if (bytes.len() as u64) > self.max_bytes.saturating_sub(self.byte_len) {
                return Err(io::Error::other("artifact output exceeds byte limit"));
            }
            let count = self.output.write(bytes)?;
            self.byte_len += count as u64;
            Ok(count)
        })();
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }

    fn flush(&mut self) -> io::Result<()> {
        let result = (|| {
            if self.failed.load(Ordering::Acquire) {
                return Err(io::Error::other("processor stream failed"));
            }
            self.cancelled.check().map_err(io::Error::other)?;
            self.output.flush()
        })();
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
}

// Invoke the standard helpers through primitive operations only, avoiding
// recursion into our overrides while retaining budget/cancellation checks.
struct PrimitiveIo<'a, T: ?Sized>(&'a mut T);

impl<T: Read + ?Sized> Read for PrimitiveIo<'_, T> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.0.read(bytes)
    }
}

impl<T: Write + ?Sized> Write for PrimitiveIo<'_, T> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

fn sticky_io<T>(failed: &AtomicBool, operation: impl FnOnce() -> io::Result<T>) -> io::Result<T> {
    struct FailureGuard<'a>(Option<&'a AtomicBool>);

    impl Drop for FailureGuard<'_> {
        fn drop(&mut self) {
            if let Some(failed) = self.0 {
                failed.store(true, Ordering::Release);
            }
        }
    }

    // Also poison on unwind: a formatter may fail without an underlying I/O
    // error, causing std's write_fmt to panic, which plugin code can catch.
    let mut guard = FailureGuard(Some(failed));
    let result = operation();
    if result.is_ok() {
        guard.0 = None;
    }
    result
}

/// A complete graph node and its immutable output specification.
pub struct PreparedProcess {
    /// Frozen streaming node with recorded upstream dependencies.
    pub job: BuildJob,
    /// Matching immutable output contract, checked again by BuildExecutor.
    pub spec: ArtifactSpec,
}

/// Analyze a trusted plugin and freeze the complete execution contract. The
/// callback is panic-isolated and failed API operations poison the plan.
pub fn prepare_process(
    processor: Arc<dyn Processor>,
    catalog: &InputCatalog<'_>,
    request: ProcessRequest,
    limits: ProcessLimits,
    cancelled: ProcessCancellation,
) -> Result<PreparedProcess, ProcessorError> {
    cancelled.check()?;
    let descriptor = catch_unwind(AssertUnwindSafe(|| processor.descriptor().clone()))
        .map_err(|_| error("processor descriptor panicked"))?;
    if descriptor.identity.id == 0
        || descriptor.identity.version == 0
        || descriptor.name.is_empty()
        || descriptor.settings_schema == 0
        || !descriptor.deterministic
        || descriptor.output_types.is_empty()
        || descriptor
            .input_types
            .iter()
            .chain(&descriptor.output_types)
            .any(|ty| ty.kind == 0 || ty.schema == 0)
    {
        return Err(error("invalid shared processor descriptor"));
    }
    for bytes in [
        &request.canonical_settings,
        &request.toolchain_fingerprint,
        &request.environment_fingerprint,
    ] {
        if bytes.len() > limits.max_settings_bytes {
            return Err(error("processor request exceeds settings limit"));
        }
    }
    let failed = Arc::new(AtomicBool::new(false));
    let mut context = AnalyzeContext {
        store: catalog.store,
        catalog: &catalog.entries,
        permitted: &descriptor.input_types,
        declared: BTreeMap::new(),
        limits,
        remaining: Arc::new(AtomicU64::new(limits.max_analysis_bytes)),
        cancelled: cancelled.clone(),
        failed: Arc::clone(&failed),
    };
    for name in &request.primary_inputs {
        context.declare_input(name)?;
    }
    let plan = catch_unwind(AssertUnwindSafe(|| {
        processor.analyze(&mut context, &request)
    }))
    .map_err(|_| error("processor analysis panicked"))??;
    cancelled.check()?;
    if failed.load(Ordering::Acquire) {
        return Err(error("processor analysis input access failed"));
    }
    if !descriptor.output_types.contains(&plan.output)
        || !plan.alignment.is_power_of_two()
        || plan.resources.cpu_threads == 0
        || plan.canonical_settings.len() > limits.max_settings_bytes
    {
        return Err(error(
            "processor plan violates output, resource, or settings capabilities",
        ));
    }
    let mut settings = b"PRISM-PROCESS-PLAN\0\x01".to_vec();
    settings.extend_from_slice(&descriptor.settings_schema.to_le_bytes());
    settings.push(descriptor.category as u8);
    for bytes in [&request.canonical_settings, &plan.canonical_settings] {
        settings.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        settings.extend_from_slice(bytes);
    }
    settings.extend_from_slice(&(context.declared.len() as u64).to_le_bytes());
    for (name, input) in &context.declared {
        settings.extend_from_slice(&(name.len() as u64).to_le_bytes());
        settings.extend_from_slice(name.as_bytes());
        settings.extend_from_slice(&input.descriptor.kind.to_le_bytes());
        settings.extend_from_slice(&input.descriptor.schema.to_le_bytes());
        settings.extend_from_slice(&input.descriptor.key.0);
        settings.extend_from_slice(&input.descriptor.content_hash.0);
        settings.extend_from_slice(&input.descriptor.byte_len.to_le_bytes());
    }
    let spec = ArtifactSpec {
        asset: request.asset,
        processor_id: descriptor.identity.id,
        processor_version: descriptor.identity.version,
        canonical_settings: settings,
        profile_hash: request.profile_hash,
        toolchain_fingerprint: request.toolchain_fingerprint,
        environment_fingerprint: request.environment_fingerprint,
        kind: plan.output.kind,
        schema: plan.output.schema,
        alignment: plan.alignment,
    };
    let names: Vec<_> = context.declared.keys().cloned().collect();
    let inputs: Vec<_> = context.declared.into_values().collect();
    if inputs.iter().any(|input| input.job == request.job) {
        return Err(error("processor cannot depend on itself"));
    }
    let dependencies: Vec<_> = inputs.iter().map(|input| input.job).collect();
    let resources = plan.resources;
    let frozen = Arc::new(FrozenProcess {
        processor,
        plan,
        names,
        inputs,
        spec: spec.clone(),
        max_output_bytes: limits.max_output_bytes,
        cancelled,
    });
    Ok(PreparedProcess {
        job: BuildJob::planned(request.job, dependencies, frozen)
            .with_resources(resources)
            .with_max_output_bytes(limits.max_output_bytes),
        spec,
    })
}

pub(crate) struct FrozenProcess {
    processor: Arc<dyn Processor>,
    plan: ProcessPlan,
    names: Vec<String>,
    inputs: Vec<DeclaredInput>,
    spec: ArtifactSpec,
    max_output_bytes: u64,
    cancelled: ProcessCancellation,
}

impl FrozenProcess {
    pub(crate) fn max_output_bytes(&self) -> u64 {
        self.max_output_bytes
    }
    pub(crate) fn cacheable(&self) -> bool {
        self.plan.cacheable
    }
    pub(crate) fn check(&self) -> Result<(), ProcessorError> {
        self.cancelled.check()
    }
    pub(crate) fn validate_admission(
        &self,
        spec: &ArtifactSpec,
        resources: BuildResources,
    ) -> Result<(), ProcessorError> {
        self.check()?;
        if spec != &self.spec {
            return Err(error("artifact spec differs from frozen processor plan"));
        }
        if resources != self.plan.resources {
            return Err(error("job resources differ from frozen processor plan"));
        }
        Ok(())
    }
    pub(crate) fn validate(
        &self,
        spec: &ArtifactSpec,
        inputs: &[ArtifactDescriptor],
    ) -> Result<(), ProcessorError> {
        self.check()?;
        if spec != &self.spec {
            return Err(error("artifact spec differs from frozen processor plan"));
        }
        if inputs.len() != self.inputs.len()
            || inputs
                .iter()
                .zip(&self.inputs)
                .any(|(actual, expected)| actual != &expected.descriptor)
        {
            return Err(error("execution inputs differ from frozen processor plan"));
        }
        Ok(())
    }

    pub(crate) fn execute(
        &self,
        inputs: &mut [ProcessorInput<'_>],
        output: &mut dyn Write,
    ) -> Result<(), ProcessorError> {
        self.check()?;
        if inputs.len() != self.inputs.len()
            || inputs.iter().zip(&self.inputs).any(|(actual, expected)| {
                actual.content_hash() != expected.descriptor.content_hash
                    || actual.byte_len() != expected.descriptor.byte_len
            })
        {
            return Err(error(
                "execution input content differs from frozen processor plan",
            ));
        }
        let failed = AtomicBool::new(false);
        let mut ctx = ExecuteContext {
            inputs,
            names: &self.names,
            cancelled: &self.cancelled,
            failed: &failed,
        };
        let mut writer = ArtifactWriter {
            output,
            cancelled: &self.cancelled,
            failed: &failed,
            byte_len: 0,
            max_bytes: self.max_output_bytes,
        };
        let report = catch_unwind(AssertUnwindSafe(|| {
            self.processor.execute(&mut ctx, &self.plan, &mut writer)
        }))
        .map_err(|_| error("processor execution panicked"))??;
        self.check()?;
        if failed.load(Ordering::Acquire) {
            return Err(error("processor execution input or output access failed"));
        }
        if report.output_bytes != writer.bytes_written() {
            return Err(error("processor report differs from output length"));
        }
        writer.flush().map_err(display_error)?;
        Ok(())
    }
}

fn charge(remaining: &AtomicU64, amount: u64) -> Result<(), ProcessorError> {
    remaining
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
            left.checked_sub(amount)
        })
        .map(|_| ())
        .map_err(|_| error("processor analysis exceeds byte budget"))
}
fn error(message: impl Into<String>) -> ProcessorError {
    ProcessorError::new(message)
}
fn display_error(error: impl std::fmt::Display) -> ProcessorError {
    ProcessorError::new(error.to_string())
}

#[cfg(test)]
mod sticky_io_tests {
    use super::*;

    struct ZeroWriter;

    impl Write for ZeroWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Ok(0)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn assert_write_zero_is_sticky(formatted: bool) {
        let failed = AtomicBool::new(false);
        let cancelled = ProcessCancellation::default();
        let mut sink = ZeroWriter;
        let mut writer = ArtifactWriter {
            output: &mut sink,
            cancelled: &cancelled,
            failed: &failed,
            byte_len: 0,
            max_bytes: 64,
        };
        let error = if formatted {
            writer.write_fmt(format_args!("value:{}", 7)).unwrap_err()
        } else {
            writer.write_all(b"payload").unwrap_err()
        };
        assert_eq!(error.kind(), io::ErrorKind::WriteZero);
        assert!(failed.load(Ordering::Acquire));
        assert_eq!(writer.bytes_written(), 0);
        assert!(writer.write(b"later").is_err());
        assert!(writer.flush().is_err());
    }

    #[test]
    fn write_all_generated_write_zero_is_sticky() {
        assert_write_zero_is_sticky(false);
    }

    #[test]
    fn write_fmt_generated_write_zero_is_sticky() {
        assert_write_zero_is_sticky(true);
    }
}
