use std::fmt;
use std::sync::Arc;

mod stream;
pub(crate) use stream::{admit_materialized_inputs, materialize_inputs};
pub use stream::{ProcessorInput, StreamingProcessor, DEFAULT_MATERIALIZED_INPUT_BYTES};

/// Stable identity used to order jobs and refer to dependencies in a build graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct JobId(pub u128);

/// Failure returned by a local processor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessorError {
    message: String,
}

impl ProcessorError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ProcessorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProcessorError {}

/// Synchronous local transformation. Inputs are passed in the exact order
/// declared by [`BuildJob::dependencies`]. A processor should be a pure
/// function of those bytes and its captured, immutable configuration: it must
/// not read ambient state or produce nondeterministic output.
///
/// Processors are shared by the serial and parallel graph executors. They must
/// be thread-safe and deterministic: stateful interior-mutability can still
/// violate reproducibility and is the implementor's responsibility to avoid.
pub trait Processor: Send + Sync {
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError>;
}

impl<F> Processor for F
where
    F: for<'a, 'b> Fn(&'a [&'b [u8]]) -> Result<Vec<u8>, ProcessorError> + Send + Sync,
{
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        self(inputs)
    }
}

/// A single output-producing graph job.
pub struct BuildJob {
    id: JobId,
    dependencies: Vec<JobId>,
    processor: JobProcessor,
    resources: crate::resources::BuildResources,
    max_output_bytes: u64,
    max_materialized_input_bytes: u64,
}

#[derive(Clone)]
pub(crate) enum JobProcessor {
    Buffered(Arc<dyn Processor>),
    Streaming(Arc<dyn StreamingProcessor>),
    Planned(Arc<crate::processor::FrozenProcess>),
}

impl JobProcessor {
    pub(crate) fn validate_admission(
        &self,
        spec: &crate::ArtifactSpec,
        resources: crate::BuildResources,
    ) -> Result<(), ProcessorError> {
        match self {
            Self::Planned(process) => process.validate_admission(spec, resources),
            _ => Ok(()),
        }
    }

    pub(crate) fn cacheable(&self) -> bool {
        match self {
            Self::Planned(process) => process.cacheable(),
            _ => true,
        }
    }

    pub(crate) fn check(&self) -> Result<(), ProcessorError> {
        match self {
            Self::Planned(process) => process.check(),
            _ => Ok(()),
        }
    }

    pub(crate) fn validate(
        &self,
        spec: &crate::ArtifactSpec,
        inputs: &[crate::ArtifactDescriptor],
    ) -> Result<(), ProcessorError> {
        match self {
            Self::Planned(process) => process.validate(spec, inputs),
            _ => Ok(()),
        }
    }
}

impl BuildJob {
    pub(crate) fn planned(
        id: JobId,
        dependencies: Vec<JobId>,
        process: Arc<crate::processor::FrozenProcess>,
    ) -> Self {
        Self {
            id,
            dependencies,
            processor: JobProcessor::Planned(process),
            resources: crate::resources::BuildResources::default(),
            max_output_bytes: u64::MAX,
            max_materialized_input_bytes: DEFAULT_MATERIALIZED_INPUT_BYTES,
        }
    }

    /// Create a job. Dependency order is semantically significant and is
    /// preserved when the processor receives its input byte slices.
    pub fn new<P>(id: JobId, dependencies: impl Into<Vec<JobId>>, processor: P) -> Self
    where
        P: Processor + 'static,
    {
        Self {
            id,
            dependencies: dependencies.into(),
            processor: JobProcessor::Buffered(Arc::new(processor)),
            resources: crate::resources::BuildResources::default(),
            max_output_bytes: u64::MAX,
            max_materialized_input_bytes: DEFAULT_MATERIALIZED_INPUT_BYTES,
        }
    }

    /// Construct a native streaming node. The persistent executor never
    /// materializes its input or output payloads in a coordinator-owned Vec.
    pub fn streaming<P: StreamingProcessor + 'static>(
        id: JobId,
        dependencies: impl Into<Vec<JobId>>,
        processor: P,
    ) -> Self {
        Self {
            id,
            dependencies: dependencies.into(),
            processor: JobProcessor::Streaming(Arc::new(processor)),
            resources: crate::resources::BuildResources::default(),
            max_output_bytes: u64::MAX,
            max_materialized_input_bytes: DEFAULT_MATERIALIZED_INPUT_BYTES,
        }
    }

    /// Aggregate unique input byte admission for contiguous-input adapters.
    /// It does not constrain trusted processor heap or native stream readers.
    /// This execution limit does not alter the semantic artifact key.
    pub fn with_max_materialized_input_bytes(mut self, max_bytes: u64) -> Self {
        self.max_materialized_input_bytes = max_bytes;
        self
    }

    pub fn max_materialized_input_bytes(&self) -> u64 {
        self.max_materialized_input_bytes
    }

    pub fn id(&self) -> JobId {
        self.id
    }

    /// Declare peak concurrent demand for resource-aware execution. This is
    /// scheduling policy and does not alter the semantic artifact key.
    pub fn with_resources(mut self, resources: crate::resources::BuildResources) -> Self {
        self.resources = resources;
        self
    }

    pub fn resources(&self) -> crate::resources::BuildResources {
        self.resources
    }

    /// Limits each encoded output, including verified cache hits and both
    /// determinism runs. Zero permits only an empty output. The default is
    /// `u64::MAX`, meaning no caller-specified bound. This does not constrain
    /// processor heap allocations or aggregate outputs, and does not change
    /// the artifact key; every cache reuse is checked against the current bound.
    pub fn with_max_output_bytes(mut self, max_output_bytes: u64) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    pub fn max_output_bytes(&self) -> u64 {
        match &self.processor {
            JobProcessor::Planned(process) => self.max_output_bytes.min(process.max_output_bytes()),
            _ => self.max_output_bytes,
        }
    }

    pub fn dependencies(&self) -> &[JobId] {
        &self.dependencies
    }

    pub(crate) fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        let output = match &self.processor {
            JobProcessor::Buffered(processor) => processor.process(inputs)?,
            JobProcessor::Streaming(processor) => {
                let mut readers: Vec<_> = inputs
                    .iter()
                    .map(|bytes| ProcessorInput::borrowed(bytes))
                    .collect();
                let mut output = stream::BufferOutput::new(self.max_output_bytes);
                processor.execute(&mut readers, &mut output)?;
                output.finish()?
            }
            JobProcessor::Planned(processor) => {
                let mut readers: Vec<_> = inputs
                    .iter()
                    .map(|bytes| ProcessorInput::borrowed(bytes))
                    .collect();
                let mut output = stream::BufferOutput::new(self.max_output_bytes);
                processor.execute(&mut readers, &mut output)?;
                output.finish()?
            }
        };
        if output.len() as u64 > self.max_output_bytes {
            return Err(ProcessorError::new(
                "processor output exceeds declared byte bound",
            ));
        }
        Ok(output)
    }

    pub(crate) fn processor(&self) -> JobProcessor {
        self.processor.clone()
    }
}
